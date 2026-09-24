"""Train ALMT score regression, then classify at -threshold and +threshold."""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
except ImportError:
    from q2_metrics_fallback import accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from train_q2_aligned import (AMP_DTYPE, LABELS, ROOT, AlignedQ2Dataset,
                              build_q2_model, set_seed)

CONFIG = ROOT / "configs" / "q2_score.yaml"


class ScoreDataset(AlignedQ2Dataset):
    def __init__(self, entry: dict, split: str):
        super().__init__(entry, split)
        self.scores = torch.from_numpy(np.asarray(
            entry["regression_labels"], dtype=np.float32).reshape(-1).copy())
        self.ids = list(entry["id"])
        if len(self.scores) != len(self.labels) or len(self.ids) != len(self.labels):
            raise ValueError(f"{split}: score and sample counts differ")

    def __getitem__(self, index: int):
        vision, audio, text, label = super().__getitem__(index)
        return vision, audio, text, label, self.scores[index], self.ids[index]


def load_score_splits(path: Path) -> dict[str, ScoreDataset]:
    with path.open("rb") as stream:
        source = pickle.load(stream)
    if set(source) != {"train", "valid", "test"}:
        raise ValueError("Expected train/valid/test in the supplied pickle")
    datasets = {split: ScoreDataset(source[split], split)
                for split in ("train", "valid", "test")}
    del source
    return datasets


def score_to_class(values: np.ndarray, threshold: float) -> np.ndarray:
    """0=negative, 1=neutral, 2=positive; boundaries belong to neutral."""
    return np.where(values < -threshold, 0, np.where(values > threshold, 2, 1))


def score_metrics(true_score: np.ndarray, pred_score: np.ndarray,
                  true_class: np.ndarray, threshold: float) -> dict:
    pred_class = score_to_class(pred_score, threshold)
    correlation = (float(np.corrcoef(true_score, pred_score)[0, 1])
                   if len(true_score) > 1 and np.std(true_score) > 0 and np.std(pred_score) > 0
                   else None)
    return {
        "accuracy": float(accuracy_score(true_class, pred_class)),
        "macro_f1": float(f1_score(true_class, pred_class, labels=[0, 1, 2],
                                   average="macro", zero_division=0)),
        "mae": float(np.mean(np.abs(true_score - pred_score))),
        "pearson": correlation,
        "recall": {name: float(value) for name, value in zip(
            LABELS, recall_score(true_class, pred_class, labels=[0, 1, 2],
                                 average=None, zero_division=0))},
        "confusion": confusion_matrix(true_class, pred_class, labels=[0, 1, 2]).tolist(),
        "threshold": threshold,
        "oracle_threshold_accuracy": float(accuracy_score(
            true_class, score_to_class(true_score, threshold))),
    }


@torch.inference_mode()
def evaluate_score(model, loader, device: torch.device, amp: bool,
                   threshold: float, max_batches: int = 0):
    model.eval()
    ids, true_scores, pred_scores, true_classes = [], [], [], []
    for batch_index, (vision, audio, text, label, score, sample_id) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            output = model(vision.to(device, non_blocking=True),
                           audio.to(device, non_blocking=True),
                           text.to(device, non_blocking=True))
        # The source ALMT regression head is unconstrained. Report on the
        # annotation scale; clipping does not change the ±0.5 decision.
        pred_scores.extend(output.squeeze(-1).float().clamp(-3, 3).cpu().tolist())
        true_scores.extend(score.tolist())
        true_classes.extend(label.tolist())
        ids.extend(sample_id)
    true_scores = np.asarray(true_scores, dtype=np.float32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)
    true_classes = np.asarray(true_classes, dtype=np.int64)
    result = score_metrics(true_scores, pred_scores, true_classes, threshold)
    rows = [{"id": sample_id, "true_score": float(ts), "pred_score": float(ps),
             "true_class": int(tc), "pred_class": int(pc)}
            for sample_id, ts, ps, tc, pc in zip(
                ids, true_scores, pred_scores, true_classes,
                score_to_class(pred_scores, threshold))]
    return result, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "id", "true_score", "pred_score", "true_class", "pred_class"))
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                amp: bool, accumulation: int, max_batches: int = 0) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, items = 0.0, 0
    batch_count = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, (vision, audio, text, _label, score, _ids) in enumerate(loader):
        if batch_index >= batch_count:
            break
        vision, audio, text = (tensor.to(device, non_blocking=True)
                               for tensor in (vision, audio, text))
        score = score.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            pred_score = model(vision, audio, text).squeeze(-1)
            loss = F.mse_loss(pred_score.float(), score)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
        group_size = min(accumulation, batch_count - (batch_index // accumulation) * accumulation)
        scaler.scale(loss / group_size).backward()
        loss_sum += float(loss.detach()) * len(score)
        items += len(score)
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                scheduler.step()
    return loss_sum / max(items, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke-batches", type=int, default=0)
    cli = parser.parse_args()
    config = yaml.safe_load(cli.config.read_text(encoding="utf-8"))
    source = Path(config["dataset"]["dataPath"])
    bert_dir = Path(config["model"]["bert_pretrained"])
    if not source.is_file() or not (bert_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Check data and BERT paths: {source}, {bert_dir}")
    threshold = float(config["task"]["neutral_threshold"])
    if not 0 < threshold < 3:
        parser.error("neutral_threshold must be in (0, 3)")
    if int(config["model"]["output_dim"]) != 1:
        parser.error("score regression requires model.output_dim=1")
    set_seed(int(config["base"]["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    output = ROOT / "runs" / (config["base"]["project_name"] +
                               ("_smoke" if cli.smoke_batches else ""))
    output.mkdir(parents=True, exist_ok=True)

    datasets = load_score_splits(source)
    batch_size = int(config["base"]["batch_size"])
    neutral_weight = float(config.get("sampling", {}).get("neutral_weight", 1.0))
    if not math.isfinite(neutral_weight) or neutral_weight < 1.0:
        parser.error("sampling.neutral_weight must be finite and >= 1.0")
    train_sampler = None
    if neutral_weight > 1.0:
        train_labels = datasets["train"].labels
        weights = torch.ones(len(train_labels), dtype=torch.double)
        weights[train_labels == 1] = neutral_weight
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(train_labels), replacement=True,
            generator=torch.Generator().manual_seed(int(config["base"]["seed"])))
        counts = torch.bincount(train_labels, minlength=3).tolist()
        weighted_total = counts[0] + neutral_weight * counts[1] + counts[2]
        print(f"train-only neutral sampling: weight={neutral_weight:g}, "
              f"expected class shares=[{counts[0] / weighted_total:.1%}, "
              f"{neutral_weight * counts[1] / weighted_total:.1%}, "
              f"{counts[2] / weighted_total:.1%}]; "
              "valid/test use original distribution", flush=True)
    loaders = {
        split: DataLoader(
            dataset, batch_size=batch_size,
            shuffle=split == "train" and train_sampler is None,
            sampler=train_sampler if split == "train" else None,
            pin_memory=device.type == "cuda",
            num_workers=int(config["base"]["num_workers"]))
        for split, dataset in datasets.items()
    }
    model = build_q2_model(config, device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.bertmodel.parameters()),
         "lr": float(config["base"]["bert_lr"])},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("bertmodel.")],
         "lr": float(config["base"]["other_lr"])},
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
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16)
    print(f"device={device} amp={amp} train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} threshold=±{threshold} output={output}", flush=True)
    print("ALMT regression: text_bert -> BERT, aligned audio/vision; "
          "classification from score; no artificial missingness; test untouched.", flush=True)

    history, best_key, best_epoch, stale = [], (-1.0, -1.0, -float("inf")), 0, 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                           device, amp, accumulation, cli.smoke_batches)
        valid, rows = evaluate_score(model, loaders["valid"], device, amp,
                                     threshold, cli.smoke_batches)
        history.append({"epoch": epoch, "train_mse": loss, "valid": valid})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        key = (valid["macro_f1"], valid["accuracy"], -valid["mae"])
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "valid": valid, "source": str(source)}, output / "best.pt")
            write_rows(output / "valid_predictions.csv", rows)
        else:
            stale += 1
        print(f"epoch={epoch:02d} mse={loss:.4f} val_acc={valid['accuracy']:.2%} "
              f"val_macro_F1={valid['macro_f1']:.3f} val_MAE={valid['mae']:.3f} "
              f"recall={valid['recall']}", flush=True)
        if stale >= int(config["base"]["patience"]) and not cli.smoke_batches:
            print(f"early_stop: best_epoch={best_epoch}", flush=True)
            break
    if cli.smoke_batches:
        print("SMOKE CHECK ONLY: these metrics are not full validation results", flush=True)
    else:
        best = max(history, key=lambda item: (item["valid"]["macro_f1"],
                                              item["valid"]["accuracy"],
                                              -item["valid"]["mae"]))
        (output / "validation_complete.json").write_text(json.dumps({
            "best_epoch": best_epoch, "best_valid": best["valid"],
            "test_evaluated": False}, indent=2), encoding="utf-8")
        print(f"validation complete: best_epoch={best_epoch}; "
              f"scores={output / 'valid_predictions.csv'}", flush=True)


if __name__ == "__main__":
    main()
