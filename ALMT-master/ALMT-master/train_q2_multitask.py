"""Train ALMT with a score head and a three-class head on aligned E-question data."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from train_q2_aligned import AMP_DTYPE, ROOT, metrics, set_seed
from train_q2_score import load_score_splits, score_metrics, score_to_class
from models.almt_multitask import ALMTMultiTask

CONFIG = ROOT / "configs" / "q2_multitask.yaml"


def build_model(config: dict, device: torch.device) -> ALMTMultiTask:
    args = SimpleNamespace(model=SimpleNamespace(**config["model"]))
    return ALMTMultiTask(args).to(device)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, amp: bool,
             threshold: float, max_batches: int = 0) -> tuple[dict, list[dict]]:
    model.eval()
    ids, actual_scores, predicted_scores, actual_classes = [], [], [], []
    predicted_classes, probabilities = [], []
    for batch_index, (vision, audio, text, label, score, sample_id) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            score_output, logits = model(
                vision.to(device, non_blocking=True),
                audio.to(device, non_blocking=True),
                text.to(device, non_blocking=True))
        ids.extend(sample_id)
        actual_scores.extend(score.tolist())
        predicted_scores.extend(score_output.squeeze(-1).float().clamp(-3, 3).cpu().tolist())
        actual_classes.extend(label.tolist())
        predicted_classes.extend(logits.argmax(dim=-1).cpu().tolist())
        probabilities.extend(F.softmax(logits.float(), dim=-1).cpu().tolist())

    y = np.asarray(actual_classes, dtype=np.int64)
    s = np.asarray(predicted_scores, dtype=np.float32)
    score_result = score_metrics(np.asarray(actual_scores, dtype=np.float32), s, y, threshold)
    class_result = metrics(actual_classes, predicted_classes)
    score_classes = score_to_class(s, threshold)
    rows = [
        {"id": sample_id, "true_score": float(actual_score),
         "pred_score": float(pred_score), "true_class": int(actual_class),
         "score_rule_class": int(score_class), "class_head_class": int(head_class),
         "prob_negative": float(probs[0]), "prob_neutral": float(probs[1]),
         "prob_positive": float(probs[2])}
        for sample_id, actual_score, pred_score, actual_class, score_class,
            head_class, probs in zip(ids, actual_scores, predicted_scores,
                                     actual_classes, score_classes,
                                     predicted_classes, probabilities)
    ]
    return {"score_rule": score_result, "class_head": class_result}, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "id", "true_score", "pred_score", "true_class", "score_rule_class",
            "class_head_class", "prob_negative", "prob_neutral", "prob_positive"))
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                amp: bool, accumulation: int, class_weight: float,
                max_batches: int = 0) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {"mse": 0.0, "ce": 0.0, "combined": 0.0}
    count = 0
    batch_count = min(len(loader), max_batches) if max_batches else len(loader)
    for batch_index, (vision, audio, text, label, score, _ids) in enumerate(loader):
        if batch_index >= batch_count:
            break
        vision, audio, text = (value.to(device, non_blocking=True)
                               for value in (vision, audio, text))
        label = label.to(device, non_blocking=True)
        score = score.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            predicted_score, logits = model(vision, audio, text)
            mse = F.mse_loss(predicted_score.squeeze(-1).float(), score)
            ce = F.cross_entropy(logits.float(), label)
            loss = mse + class_weight * ce
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
        group_size = min(accumulation, batch_count - (batch_index // accumulation) * accumulation)
        scaler.scale(loss / group_size).backward()
        for key, value in (("mse", mse), ("ce", ce), ("combined", loss)):
            sums[key] += float(value.detach()) * len(score)
        count += len(score)
        if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_count:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                scheduler.step()
    return {key: value / max(count, 1) for key, value in sums.items()}


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
    class_weight = float(config["task"]["classification_loss_weight"])
    if not 0 < threshold < 3 or not math.isfinite(class_weight) or class_weight <= 0:
        parser.error("Use neutral_threshold in (0,3) and positive classification_loss_weight")
    set_seed(int(config["base"]["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    output = ROOT / "runs" / (config["base"]["project_name"] +
                               ("_smoke" if cli.smoke_batches else ""))
    output.mkdir(parents=True, exist_ok=True)

    datasets = load_score_splits(source)
    loaders = {
        split: DataLoader(dataset, batch_size=int(config["base"]["batch_size"]),
                          shuffle=split == "train", pin_memory=device.type == "cuda",
                          num_workers=int(config["base"]["num_workers"]))
        for split, dataset in datasets.items()
    }
    model = build_model(config, device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.bertmodel.parameters()), "lr": float(config["base"]["bert_lr"])},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("bertmodel.")],
         "lr": float(config["base"]["other_lr"])}
    ], weight_decay=float(config["base"]["weight_decay"]))
    accumulation = int(config["base"]["accumulation_steps"])
    epochs = 1 if cli.smoke_batches else int(config["base"]["n_epochs"])
    batches = min(len(loaders["train"]), cli.smoke_batches) if cli.smoke_batches else len(loaders["train"])
    total_steps = epochs * math.ceil(batches / accumulation)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(0.1, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and AMP_DTYPE == torch.float16)
    print(f"device={device} amp={amp} train={len(datasets['train'])} "
          f"valid={len(datasets['valid'])} threshold=±{threshold} "
          f"classification_loss_weight={class_weight} output={output}", flush=True)
    print("ALMT multi-task: text_bert, aligned audio/vision; no neutral oversampling; "
          "test untouched.", flush=True)

    history = []
    best_score_key = (-1.0, -1.0, -float("inf"))
    best_class_key = (-1.0, -1.0)
    best_score_epoch = best_class_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, scheduler,
                                 scaler, device, amp, accumulation, class_weight,
                                 cli.smoke_batches)
        valid, rows = evaluate(model, loaders["valid"], device, amp,
                                threshold, cli.smoke_batches)
        history.append({"epoch": epoch, "train": train_loss, "valid": valid})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        score_key = (valid["score_rule"]["macro_f1"],
                     valid["score_rule"]["accuracy"], -valid["score_rule"]["mae"])
        class_key = (valid["class_head"]["macro_f1"], valid["class_head"]["accuracy"])
        saved = {"state_dict": model.state_dict(), "config": config,
                 "epoch": epoch, "valid": valid, "source": str(source)}
        score_improved = score_key > best_score_key
        class_improved = class_key > best_class_key
        if score_improved:
            best_score_key, best_score_epoch = score_key, epoch
            torch.save(saved, output / "best_score.pt")
            write_rows(output / "valid_predictions_score_best.csv", rows)
        if class_improved:
            best_class_key, best_class_epoch = class_key, epoch
            torch.save(saved, output / "best_class.pt")
            write_rows(output / "valid_predictions_class_best.csv", rows)
        stale = 0 if score_improved or class_improved else stale + 1
        print(f"epoch={epoch:02d} mse={train_loss['mse']:.4f} ce={train_loss['ce']:.4f} "
              f"score_acc={valid['score_rule']['accuracy']:.2%} "
              f"score_F1={valid['score_rule']['macro_f1']:.3f} "
              f"score_neutral_recall={valid['score_rule']['recall']['Neutral']:.3f} "
              f"class_acc={valid['class_head']['accuracy']:.2%} "
              f"class_F1={valid['class_head']['macro_f1']:.3f} "
              f"class_neutral_recall={valid['class_head']['recall']['Neutral']:.3f}",
              flush=True)
        if stale >= int(config["base"]["patience"]) and not cli.smoke_batches:
            print(f"early_stop: score_epoch={best_score_epoch}, "
                  f"class_epoch={best_class_epoch}", flush=True)
            break
    if cli.smoke_batches:
        print("SMOKE CHECK ONLY: partial validation metrics are not experiment results", flush=True)
    else:
        summary = {"best_score_epoch": best_score_epoch,
                   "best_class_epoch": best_class_epoch,
                   "best_score_checkpoint_valid": history[best_score_epoch - 1]["valid"],
                   "best_class_checkpoint_valid": history[best_class_epoch - 1]["valid"],
                   "test_evaluated": False}
        (output / "validation_complete.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"validation complete: score epoch={best_score_epoch}; "
              f"class epoch={best_class_epoch}; test untouched", flush=True)


if __name__ == "__main__":
    main()
