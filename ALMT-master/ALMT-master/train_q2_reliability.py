"""Train a mask-aware text-anchored model: polarity primary, intensity auxiliary."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from train_q2_aligned import AMP_DTYPE, ROOT, metrics, set_seed
from train_q2_score import load_score_splits, score_metrics, score_to_class
from models.q2_reliability import Q2ReliabilityModel

CONFIG = ROOT / "configs" / "q2_reliability.yaml"


def feature_stats(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-feature mean/std using only nonzero training frames."""
    mask = values.abs().sum(dim=-1) > 0
    count = mask.sum().clamp_min(1)
    mean = (values * mask.unsqueeze(-1)).sum(dim=(0, 1)) / count
    variance = (((values - mean) ** 2) * mask.unsqueeze(-1)).sum(dim=(0, 1)) / count
    return mean, variance.sqrt().clamp_min(1e-3)


def build_model(config: dict, train_dataset, device: torch.device) -> Q2ReliabilityModel:
    audio_mean, audio_std = feature_stats(train_dataset.audio)
    vision_mean, vision_std = feature_stats(train_dataset.vision)
    return Q2ReliabilityModel(config, audio_mean, audio_std,
                              vision_mean, vision_std).to(device)


def check_word_alignment(dataset, split: str) -> None:
    """Reject feature files that lack the token-position layout required by MAG."""
    token_lengths = dataset.text[:, 1, :].bool().sum(dim=1)
    positions = torch.arange(dataset.audio.shape[1]).unsqueeze(0)
    words = (positions > 0) & (positions < token_lengths.unsqueeze(1) - 1)
    audio_observed = dataset.audio.abs().sum(dim=-1) > 0
    vision_observed = dataset.vision.abs().sum(dim=-1) > 0
    matched = (audio_observed.sum(dim=1) == token_lengths - 2).float().mean().item()
    nonword = int(((audio_observed | vision_observed) & ~words).sum().item())
    print(f"{split} token-alignment check: audio_length_match={matched:.2%} "
          f"nonword_observations={nonword}", flush=True)
    if matched < 0.98 or nonword:
        raise ValueError(f"{split}: expected word-position aligned audio/vision features")


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, amp: bool,
             threshold: float, max_batches: int = 0) -> tuple[dict, list[dict]]:
    model.eval()
    ids, true_classes, true_scores, predicted_scores, predicted_classes = [], [], [], [], []
    probabilities = []
    for batch_index, (vision, audio, text, label, score, sample_id) in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
            outputs = model(
                vision.to(device, non_blocking=True),
                audio.to(device, non_blocking=True),
                text.to(device, non_blocking=True))
            logits, score_output = outputs[:2]
        ids.extend(sample_id)
        true_classes.extend(label.tolist())
        true_scores.extend(score.tolist())
        predicted_scores.extend(score_output.float().cpu().tolist())
        predicted_classes.extend(logits.argmax(dim=-1).cpu().tolist())
        probabilities.extend(F.softmax(logits.float(), dim=-1).cpu().tolist())
    y = np.asarray(true_classes, dtype=np.int64)
    s = np.asarray(predicted_scores, dtype=np.float32)
    result = {
        "class_head": metrics(true_classes, predicted_classes),
        "score_rule": score_metrics(np.asarray(true_scores, dtype=np.float32), s, y, threshold),
    }
    score_classes = score_to_class(s, threshold)
    rows = [
        {"id": sample_id, "true_class": int(label), "true_score": float(true_score),
         "class_head_class": int(head_class), "pred_score": float(pred_score),
         "score_rule_class": int(score_class),
         "prob_negative": float(probs[0]), "prob_neutral": float(probs[1]),
         "prob_positive": float(probs[2])}
        for sample_id, label, true_score, head_class, pred_score, score_class, probs
        in zip(ids, true_classes, true_scores, predicted_classes,
               predicted_scores, score_classes, probabilities)
    ]
    return result, rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "id", "true_class", "true_score", "class_head_class", "pred_score",
            "score_rule_class", "prob_negative", "prob_neutral", "prob_positive"))
        writer.writeheader()
        writer.writerows(rows)


def train_epoch(model, loader, optimizer, scheduler, scaler, device,
                amp: bool, accumulation: int, class_weights: torch.Tensor,
                label_smoothing: float, score_weight: float,
                neutral_weight: float, neutral_pos_weight: torch.Tensor,
                decision_weight: float,
                max_batches: int = 0) -> dict:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums = {"ce": 0.0, "huber": 0.0, "neutral_bce": 0.0,
            "expert_ce": 0.0, "combined": 0.0}
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
            outputs = model(vision, audio, text)
            logits, predicted_score = outputs[:2]
            ce = F.cross_entropy(logits.float(), label, weight=class_weights,
                                 label_smoothing=label_smoothing)
            huber = F.smooth_l1_loss(predicted_score.float(), score, beta=0.5)
            neutral_bce = torch.zeros((), device=device)
            if neutral_weight > 0:
                if len(outputs) != 3:
                    raise ValueError("Neutral auxiliary loss requires a neutral head")
                neutral_bce = F.binary_cross_entropy_with_logits(
                    outputs[2].float(), (label == 1).float(),
                    pos_weight=neutral_pos_weight)
            expert_ce = torch.zeros((), device=device)
            if decision_weight > 0:
                if len(outputs) != 4:
                    raise ValueError("Decision fusion loss requires text and full experts")
                expert_ce = 0.5 * (
                    F.cross_entropy(outputs[2].float(), label, weight=class_weights,
                                    label_smoothing=label_smoothing) +
                    F.cross_entropy(outputs[3].float(), label, weight=class_weights,
                                    label_smoothing=label_smoothing))
            loss = (ce + score_weight * huber + neutral_weight * neutral_bce +
                    decision_weight * expert_ce)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss in batch {batch_index}")
        group_size = min(accumulation, batch_count - (batch_index // accumulation) * accumulation)
        scaler.scale(loss / group_size).backward()
        for key, value in (("ce", ce), ("huber", huber),
                           ("neutral_bce", neutral_bce), ("expert_ce", expert_ce),
                           ("combined", loss)):
            sums[key] += float(value.detach()) * len(label)
        count += len(label)
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
    score_weight = float(config["task"]["score_loss_weight"])
    neutral_weight = float(config["task"].get("neutral_aux_loss_weight", 0.0))
    decision_weight = float(config["task"].get("decision_aux_loss_weight", 0.0))
    smoothing = float(config["task"]["label_smoothing"])
    if (not 0 < threshold < 3 or not math.isfinite(score_weight) or score_weight < 0 or
            not math.isfinite(neutral_weight) or neutral_weight < 0 or
            not math.isfinite(decision_weight) or decision_weight < 0):
        parser.error("Invalid threshold or auxiliary loss weight")
    if neutral_weight > 0 and not config["model"].get("neutral_auxiliary_head", False):
        parser.error("neutral_aux_loss_weight requires model.neutral_auxiliary_head")
    if decision_weight > 0 and not config["model"].get("decision_fusion", False):
        parser.error("decision_aux_loss_weight requires model.decision_fusion")
    if not 0 <= smoothing < 1:
        parser.error("label_smoothing must be in [0,1)")
    set_seed(int(config["base"]["seed"]))
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    output = ROOT / "runs" / (config["base"]["project_name"] +
                               ("_smoke" if cli.smoke_batches else ""))
    output.mkdir(parents=True, exist_ok=True)
    datasets = load_score_splits(source)
    if config["model"].get("word_aligned_mag", False):
        for split in ("train", "valid"):
            check_word_alignment(datasets[split], split)
    loaders = {
        split: DataLoader(dataset, batch_size=int(config["base"]["batch_size"]),
                          shuffle=split == "train", pin_memory=device.type == "cuda",
                          num_workers=int(config["base"]["num_workers"]))
        for split, dataset in datasets.items()
    }
    counts = torch.bincount(datasets["train"].labels, minlength=3).float()
    class_weights = ((counts.sum() / 3) / counts).sqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)
    neutral_pos_weight = ((counts[0] + counts[2]) / counts[1]).sqrt().to(device)
    model = build_model(config, datasets["train"], device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.bertmodel.parameters() if p.requires_grad],
         "lr": float(config["base"]["bert_lr"])},
        {"params": [p for name, p in model.named_parameters()
                    if not name.startswith("bertmodel.") and p.requires_grad],
         "lr": float(config["base"]["other_lr"])}
    ], weight_decay=float(config["base"]["weight_decay"]))
    accumulation = int(config["base"]["accumulation_steps"])
    epochs = 1 if cli.smoke_batches else int(config["base"]["n_epochs"])
    early_stopping = bool(config["base"].get("early_stopping", True))
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
          f"valid={len(datasets['valid'])} class_weights={class_weights.cpu().tolist()} "
          f"output={output}", flush=True)
    print(f"neutral_aux_loss_weight={neutral_weight} "
          f"neutral_pos_weight={neutral_pos_weight.item():.3f}", flush=True)
    print(f"decision_aux_loss_weight={decision_weight}", flush=True)
    print(f"epochs={epochs} early_stopping={early_stopping}", flush=True)
    print("Class prediction is primary; score is auxiliary. Nonzero audio/vision frames "
          "are normalized from training data; no artificial missingness; test untouched.",
          flush=True)
    history, best_key, best_epoch, stale = [], (-1.0, -1.0, -float("inf")), 0, 0
    for epoch in range(1, epochs + 1):
        train = train_epoch(model, loaders["train"], optimizer, scheduler, scaler,
                            device, amp, accumulation, class_weights, smoothing,
                            score_weight, neutral_weight, neutral_pos_weight,
                            decision_weight,
                            cli.smoke_batches)
        valid, rows = evaluate(model, loaders["valid"], device, amp,
                                threshold, cli.smoke_batches)
        history.append({"epoch": epoch, "train": train, "valid": valid})
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        key = (valid["class_head"]["macro_f1"], valid["class_head"]["accuracy"],
               -valid["score_rule"]["mae"])
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "valid": valid, "source": str(source)},
                       output / "best.pt")
            write_rows(output / "valid_predictions.csv", rows)
        else:
            stale += 1
        print(f"epoch={epoch:02d} ce={train['ce']:.4f} huber={train['huber']:.4f} "
              f"neutral_bce={train['neutral_bce']:.4f} "
              f"expert_ce={train['expert_ce']:.4f} "
              f"class_acc={valid['class_head']['accuracy']:.2%} "
              f"class_F1={valid['class_head']['macro_f1']:.3f} "
              f"neutral_recall={valid['class_head']['recall']['Neutral']:.3f} "
              f"score_MAE={valid['score_rule']['mae']:.3f} "
              f"score_Pearson={valid['score_rule']['pearson']} ", flush=True)
        if (early_stopping and stale >= int(config["base"]["patience"])
                and not cli.smoke_batches):
            print(f"early_stop: best_epoch={best_epoch}", flush=True)
            break
    if cli.smoke_batches:
        print("SMOKE CHECK ONLY: partial validation metrics are not experiment results", flush=True)
    else:
        summary = {"best_epoch": best_epoch, "best_valid": history[best_epoch - 1]["valid"],
                   "test_evaluated": False}
        (output / "validation_complete.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"validation complete: best_epoch={best_epoch}; test untouched", flush=True)


if __name__ == "__main__":
    main()
