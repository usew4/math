"""Adapt the upstream two-stage EBMC model to the competition's unaligned Q2 data.

Run from either directory with: python EBMC/train_q2.py
The sibling e_problem2 preprocessing supplies independently masked 50/100/100
sequences. Padding text to 100 positions is a tensor requirement, not alignment.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]  # e_problem2, beside EBMC-main
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from data_unaligned import DEFAULT_SOURCE, load_data  # noqa: E402
from train_unaligned import MODALITIES, scores  # noqa: E402
from ebmc import build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hide_block(batch: dict[str, torch.Tensor], *, modality: str | None = None,
               fraction: float | None = None, position: float | None = None,
               rng: random.Random | None = None) -> dict[str, torch.Tensor]:
    """Hide a continuous observed-time interval; retain original padding masks."""
    rng = rng or random
    result = {key: value.clone() for key, value in batch.items()}
    for index in range(len(result["text"])):
        choices = [name for name in MODALITIES
                   if bool(result[f"{name}_mask"][index].any())]
        if not choices or (modality is not None and modality not in choices):
            continue
        chosen = modality if modality is not None else rng.choice(choices)
        observed = torch.nonzero(result[f"{chosen}_mask"][index]).flatten()
        length = int(observed[-1]) + 1
        rate = fraction if fraction is not None else rng.uniform(.2, .6)
        width = min(length, max(1, round(length * rate)))
        start = (round((length - width) * position) if position is not None
                 else rng.randint(0, length - width))
        result[chosen][index, start:start + width] = 0
        result[f"{chosen}_mask"][index, start:start + width] = False
    return result


def ebmc_inputs(batch: dict[str, torch.Tensor], device: torch.device):
    """Pack independent streams for EBMC without assigning word/frame matches."""
    b = batch["text"].shape[0]
    audio = batch["audio"].to(device)
    vision = batch["vision"].to(device)
    text = F.pad(batch["text"].to(device), (0, 0, 0, audio.shape[1] - batch["text"].shape[1]))
    am = batch["audio_mask"].to(device).bool()
    vm = batch["vision_mask"].to(device).bool()
    tm = F.pad(batch["text_mask"].to(device).bool(), (0, audio.shape[1] - batch["text_mask"].shape[1]))
    availability = torch.stack([am.any(1), tm.any(1), vm.any(1)], dim=1)
    features = torch.cat([audio, text, vision], dim=-1).transpose(0, 1)
    frame_mask = torch.stack([am, tm, vm], dim=-1).transpose(0, 1).float()
    union_mask = (am | tm | vm).float()
    assert features.shape == (100, b, 74 + 768 + 35)
    return features, frame_mask, union_mask, availability


def masked_unimodal_ce(out, label: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    """Exclude a missing modality from its own supervised loss."""
    losses = []
    for index, logits in enumerate(out[2:5]):
        mask = availability[:, index]
        item = F.cross_entropy(logits[:, 0], label, reduction="none")
        losses.append((item * mask).sum() / mask.sum().clamp_min(1))
    return torch.stack(losses).sum() / availability.any(0).sum().clamp_min(1)


def train_epoch(args, model, loader, optimizer, stage: int) -> float:
    model.train()
    total = 0.0
    for batch_index, batch in enumerate(loader):
        if args.smoke_batches and batch_index >= args.smoke_batches:
            break
        if not args.no_robust and random.random() < args.robust_prob:
            batch = hide_block(batch)
        features, mask, union, availability = ebmc_inputs(batch, args.device)
        label = batch["class_label"].to(args.device)
        regression = batch["reg_label"].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        out = model(features, mask, union, first_stage=stage == 1, label=label,
                    do_cf=args.lambda_cce > 0, availability=availability)
        auxiliary = masked_unimodal_ce(out, label, availability)
        losses = out[5]
        if stage == 1:
            loss = auxiliary + args.lambda_msd * losses["disentangle"] + args.lambda_cce * losses["cfd"]
        else:
            classification = F.cross_entropy(out[1][:, 0], label)
            intensity = model.nlp_reg_head(out[0][:, 0]).squeeze(-1).tanh() * 3
            regression_loss = F.smooth_l1_loss(intensity, regression, beta=.5)
            loss = (classification + .5 * regression_loss + args.aux_weight * auxiliary
                    + args.lambda_msd * losses["disentangle"]
                    + args.lambda_cce * losses["cfd"]
                    + args.lambda_emc * losses["emc"]
                    + args.lambda_imtd * losses["imtd"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at stage {stage}, batch {batch_index}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, min(len(loader), args.smoke_batches or len(loader)))


@torch.inference_mode()
def evaluate(args, model, loader, scenario=None, position=.35):
    model.eval()
    true_c, pred_c, true_r, pred_r = [], [], [], []
    for batch_index, batch in enumerate(loader):
        if args.smoke_batches and batch_index >= args.smoke_batches:
            break
        if scenario is not None:
            batch = hide_block(batch, modality=scenario[0], fraction=scenario[1],
                               position=position)
        features, mask, union, availability = ebmc_inputs(batch, args.device)
        out = model(features, mask, union, first_stage=False, label=None,
                    do_cf=False, availability=availability)
        logits = out[1][:, 0]
        intensity = model.nlp_reg_head(out[0][:, 0]).squeeze(-1).tanh() * 3
        true_c.append(batch["class_label"].numpy())
        pred_c.append(logits.argmax(-1).cpu().numpy())
        true_r.append(batch["reg_label"].numpy())
        pred_r.append(intensity.cpu().numpy())
    return scores(*(np.concatenate(x) for x in (true_c, pred_c, true_r, pred_r)))


def make_args(parsed, device: torch.device):
    """Supply the original EBMC constructor's fields and Q2 task switches."""
    parsed.dataset = "CMUMOSEI"
    parsed.n_classes = 3
    parsed.n_speakers = 2
    parsed.no_cuda = device.type != "cuda"
    parsed.device = device
    parsed.test_condition = "atv"
    parsed.frame_seq = True
    parsed.q2_classification = True
    parsed.teacher_path = str(parsed.output / "stage1_teacher.pt")
    return parsed


def checkpoint(model, args, scales, epoch: int) -> dict:
    return {
        "state_dict": model.state_dict(), "scales": scales, "epoch": epoch,
        "config": {k: getattr(args, k) for k in
                   ("hidden", "depth", "num_heads", "drop_rate", "attn_drop_rate")},
        "source": str(args.source), "text_cache": str(args.text_cache),
        "task": "CMUMOSEI_Q2_3class_regression", "seed": args.seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--text-cache", type=Path,
                        default=PROJECT / "runs" / "text_bert" / "features.npz")
    parser.add_argument("--output", type=Path, default=PROJECT / "runs" / "ebmc_q2")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Total epochs, including Stage I and Stage II")
    parser.add_argument("--stage-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=8,
                        help="Stage II validation rounds without improvement before stopping")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--drop-rate", type=float, default=.15)
    parser.add_argument("--attn-drop-rate", type=float, default=0.)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--robust-prob", type=float, default=.7)
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=.2)
    parser.add_argument("--lambda-msd", type=float, default=.1)
    parser.add_argument("--lambda-cce", type=float, default=.1)
    parser.add_argument("--lambda-emc", type=float, default=.1)
    parser.add_argument("--lambda-imtd", type=float, default=.1)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    if args.stage_epochs < 1 or args.epochs <= args.stage_epochs:
        parser.error("require 1 <= --stage-epochs < --epochs")
    if args.patience < 1:
        parser.error("--patience must be positive")
    if not args.source.is_file():
        parser.error(f"missing source: {args.source}")
    if not args.text_cache.is_file():
        parser.error(f"missing text_bert cache: {args.text_cache}; run prepare_text_bert.py")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = make_args(args, device)
    datasets, scales = load_data(args.source, args.text_cache)
    training = datasets["train"]
    validation = datasets["valid"]
    testing = datasets["test"]
    if args.smoke_batches:
        count = args.smoke_batches * args.batch_size
        training = Subset(training, range(min(len(training), count)))
        validation = Subset(validation, range(min(len(validation), count)))
        testing = Subset(testing, range(min(len(testing), count)))
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(validation, batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(testing, batch_size=args.batch_size, num_workers=0)
    model = build_model(args, 74, 768, 35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    history = []
    best_score, best_epoch, patience = -1e9, 0, 0
    for epoch in range(1, args.epochs + 1):
        stage = 1 if epoch <= args.stage_epochs else 2
        loss = train_epoch(args, model, train_loader, optimizer, stage)
        if epoch == args.stage_epochs:
            torch.save(model.state_dict(), args.teacher_path)
        record = {"epoch": epoch, "stage": stage, "train_loss": loss}
        if stage == 2:
            clean = evaluate(args, model, valid_loader)
            missing = {name: evaluate(args, model, valid_loader, (name, .5))
                       for name in MODALITIES}
            robust_acc = float(np.mean([m["accuracy"] for m in missing.values()]))
            robust_f1 = float(np.mean([m["macro_f1"] for m in missing.values()]))
            robust_mae = float(np.mean([m["mae"] for m in missing.values()]))
            selection = (.5 * clean["macro_f1"] + .5 * robust_f1
                         - .05 * (clean["mae"] + robust_mae))
            record.update(valid_clean=clean, valid_missing_half=missing,
                          selection_score=selection)
            if selection > best_score + 1e-4:
                best_score, best_epoch, patience = selection, epoch, 0
                torch.save(checkpoint(model, args, scales, epoch), args.output / "best.pt")
            else:
                patience += 1
            print(f"epoch {epoch:02d} stage=2 loss={loss:.4f} "
                  f"val_acc={clean['accuracy']:.2%} val_F1={clean['macro_f1']:.3f} "
                  f"missing_acc={robust_acc:.2%} missing_F1={robust_f1:.3f}", flush=True)
        else:
            print(f"epoch {epoch:02d} stage=1 loss={loss:.4f}", flush=True)
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if stage == 2 and patience >= args.patience and not args.smoke_batches:
            break
    if args.smoke_batches:
        print("SMOKE RUN ONLY; test split not evaluated")
        return
    if args.validation_only:
        print(f"VALIDATION ONLY; best epoch {best_epoch}; test split not evaluated")
        return
    saved = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    result = {"best_epoch": best_epoch, "validation_selection_score": best_score,
              "test_clean": evaluate(args, model, test_loader)}
    if not args.skip_grid:
        result["test_missing_grid"] = [
            {"modality": name, "fraction": rate, "position": pos,
             **evaluate(args, model, test_loader, (name, rate), position=pos)}
            for name in MODALITIES for rate in (.3, .6) for pos in (0., .5, 1.)
        ]
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
