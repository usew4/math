"""One-click ALMT training on E题附件2 aligned_50.pkl (three sentiment classes)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
if importlib.util.find_spec("transformers") is None:
    sys.path.insert(0, str(ROOT / ".deps"))
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
except ImportError:
    from q2_metrics_fallback import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset

from models.almt import build_model

LABELS = ("Negative", "Neutral", "Positive")
CONFIG = ROOT / "configs" / "q2_aligned.yaml"
AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


class AlignedQ2Dataset(Dataset):
    def __init__(self, entry: dict, split: str):
        self.split = split
        self.text = torch.from_numpy(np.array(entry["text_bert"], dtype=np.int64, copy=True))
        self.audio = torch.from_numpy(np.nan_to_num(
            np.asarray(entry["audio"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy())
        self.vision = torch.from_numpy(np.nan_to_num(
            np.asarray(entry["vision"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy())
        self.labels = torch.from_numpy(np.asarray(
            entry["classification_labels"], dtype=np.int64).reshape(-1).copy())
        if self.text.shape[1:] != (3, 50) or self.audio.shape[1:] != (50, 74) or self.vision.shape[1:] != (50, 35):
            raise ValueError(f"{split}: expected text (3,50), audio (50,74), vision (50,35)")
        if not (len(self.text) == len(self.audio) == len(self.vision) == len(self.labels)):
            raise ValueError(f"{split}: sample counts differ")
        if torch.any((self.labels < 0) | (self.labels > 2)):
            raise ValueError(f"{split}: labels must be 0/1/2")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.vision[index], self.audio[index], self.text[index], self.labels[index]


def load_splits(path: Path) -> dict[str, AlignedQ2Dataset]:
    # The competition pickle is trusted local input.
    with path.open("rb") as stream:
        source = pickle.load(stream)
    if set(source) != {"train", "valid", "test"}:
        raise ValueError("Expected train/valid/test in the supplied pickle")
    datasets = {split: AlignedQ2Dataset(source[split], split)
                for split in ("train", "valid", "test")}
    del source
    return datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_q2_model(config: dict, device: torch.device):
    namespace = SimpleNamespace(model=SimpleNamespace(**config["model"]))
    return build_model(namespace).to(device)


def metrics(true: list[int], pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall": {name: float(value) for name, value in zip(
            LABELS, recall_score(true, pred, labels=[0, 1, 2], average=None, zero_division=0))},
        "confusion": confusion_matrix(true, pred, labels=[0, 1, 2]).tolist(),
    }


@torch.inference_mode()
def evaluate(model, loader, device, amp: bool, max_batches: int = 0) -> dict:
    model.eval()
    true, pred = [], []
    for batch_index, (vision, audio, text, label) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            logits = model(vision.to(device, non_blocking=True),
                           audio.to(device, non_blocking=True),
                           text.to(device, non_blocking=True))
        pred.extend(logits.argmax(dim=1).cpu().tolist())
        true.extend(label.tolist())
    return metrics(true, pred)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, amp: bool,
                accumulation: int, max_batches: int = 0) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_items = 0.0, 0
    batch_count = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, (vision, audio, text, label) in enumerate(loader):
        if batch_index >= batch_count:
            break
        vision = vision.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        text = text.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            logits = model(vision, audio, text)
            loss = F.cross_entropy(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
        group_size = min(accumulation, batch_count - (batch_index // accumulation) * accumulation)
        scaler.scale(loss / group_size).backward()
        total_loss += float(loss.detach()) * len(label)
        total_items += len(label)
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                scheduler.step()
    return total_loss / max(total_items, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Run one or more batches to check the setup; no result is valid")
    cli = parser.parse_args()
    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    source = Path(config["dataset"]["dataPath"])
    bert_dir = Path(config["model"]["bert_pretrained"])
    if not source.is_file() or not (bert_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Check data and BERT paths: {source}, {bert_dir}")
    set_seed(int(config["base"]["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    run_name = config["base"]["project_name"] + ("_smoke" if cli.smoke_batches else "")
    output = ROOT / "runs" / run_name
    output.mkdir(parents=True, exist_ok=True)

    datasets = load_splits(source)
    batch_size = int(config["base"]["batch_size"])
    loaders = {split: DataLoader(dataset, batch_size=batch_size,
                                 shuffle=split == "train", pin_memory=device.type == "cuda",
                                 num_workers=int(config["base"]["num_workers"]))
               for split, dataset in datasets.items()}
    model = build_q2_model(config, device)
    bert_parameters = list(model.bertmodel.parameters())
    other_parameters = [p for name, p in model.named_parameters()
                        if not name.startswith("bertmodel.")]
    optimizer = torch.optim.AdamW([
        {"params": bert_parameters, "lr": float(config["base"]["bert_lr"])},
        {"params": other_parameters, "lr": float(config["base"]["other_lr"])},
    ], weight_decay=float(config["base"]["weight_decay"]))
    accumulation = int(config["base"]["accumulation_steps"])
    epochs = 1 if cli.smoke_batches else int(config["base"]["n_epochs"])
    batches_per_epoch = min(len(loaders["train"]), cli.smoke_batches) if cli.smoke_batches else len(loaders["train"])
    total_steps = epochs * math.ceil(batches_per_epoch / accumulation)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(0.1, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16)
    class_counts = np.bincount(datasets["train"].labels.numpy(), minlength=3).tolist()
    print(f"device={device} amp={amp} amp_dtype={AMP_DTYPE if amp else 'off'} "
          f"train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} test={len(datasets['test'])} "
          f"classes={class_counts} output={output}", flush=True)
    print("ALMT input: text_bert(3,50) -> BERT; audio(50,74); vision(50,35). "
          "No artificial missingness. Validation selects the checkpoint; test is untouched.", flush=True)

    history, best_score, best_epoch, stale = [], (-1.0, -1.0), 0, 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                           device, amp, accumulation, cli.smoke_batches)
        valid = evaluate(model, loaders["valid"], device, amp, cli.smoke_batches)
        history.append({"epoch": epoch, "train_loss": loss, "valid": valid})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score = (valid["macro_f1"], valid["accuracy"])
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "valid": valid, "source": str(source)}, output / "best.pt")
        else:
            stale += 1
        print(f"epoch={epoch:02d} loss={loss:.4f} "
              f"val_acc={valid['accuracy']:.2%} val_macro_F1={valid['macro_f1']:.3f} "
              f"recall={valid['recall']}", flush=True)
        if stale >= int(config["base"]["patience"]) and not cli.smoke_batches:
            print(f"early_stop: best_epoch={best_epoch}", flush=True)
            break
    if cli.smoke_batches:
        print("SMOKE CHECK ONLY: these metrics are not full validation results", flush=True)
    else:
        (output / "validation_complete.json").write_text(json.dumps({
            "best_epoch": best_epoch, "best_valid": max(history, key=lambda item: (
                item["valid"]["macro_f1"], item["valid"]["accuracy"]))["valid"],
            "test_evaluated": False}, indent=2), encoding="utf-8")
        print(f"validation complete: best_epoch={best_epoch}; checkpoint={output / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
