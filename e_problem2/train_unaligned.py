"""Train and evaluate a robust unaligned three-modal sentiment model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from data_unaligned import DEFAULT_SOURCE, load_data
from model_unaligned import UnalignedFusion


ROOT = Path(__file__).resolve().parent
MODALITIES = ("text", "audio", "vision")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def missing_block(batch: dict[str, torch.Tensor], *, modality: str | None = None,
                  fraction: float | None = None, start_fraction: float | None = None,
                  rng: random.Random | None = None) -> dict[str, torch.Tensor]:
    """Hide one continuous interval inside one modality; preserve padding masks."""
    rng = rng or random
    result = {key: value.clone() for key, value in batch.items()}
    for i in range(len(result["text"])):
        options = [name for name in MODALITIES if bool(result[f"{name}_mask"][i].any())]
        if not options:
            continue
        chosen = modality if modality in options else rng.choice(options)
        observed = torch.nonzero(result[f"{chosen}_mask"][i], as_tuple=False).flatten().tolist()
        if not observed:
            continue
        length = max(observed) + 1
        rate = fraction if fraction is not None else rng.uniform(.2, .6)
        width = min(length, max(1, round(length * rate)))
        start = (min(length-width, max(0, round((length-width)*start_fraction)))
                 if start_fraction is not None else rng.randint(0, length-width))
        result[chosen][i, start:start+width] = 0
        result[f"{chosen}_mask"][i, start:start+width] = False
    return result


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def unimodal_loss(aux: dict[str, torch.Tensor], batch: dict[str, torch.Tensor],
                  ce_each: nn.Module, reg_each: nn.Module) -> torch.Tensor:
    """Average auxiliary task losses only over observed modality/sample pairs."""
    available = aux["availability"]
    per_modality = []
    for index in range(len(MODALITIES)):
        mask = available[:, index]
        class_loss = ce_each(aux["logits"][:, index], batch["class_label"])
        intensity_loss = reg_each(aux["intensity"][:, index], batch["reg_label"])
        per_modality.append(((class_loss + .5 * intensity_loss) * mask).sum()
                            / mask.sum().clamp_min(1))
    active_modalities = available.any(dim=0).sum().clamp_min(1)
    return torch.stack(per_modality).sum() / active_modalities


def scores(truth_c: np.ndarray, pred_c: np.ndarray,
           truth_r: np.ndarray, pred_r: np.ndarray) -> dict[str, float]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (truth_c, pred_c), 1)
    f1 = []
    for c in range(3):
        tp = confusion[c, c]
        precision = tp / max(confusion[:, c].sum(), 1)
        recall = tp / max(confusion[c].sum(), 1)
        f1.append(2*precision*recall/max(precision+recall, 1e-12))
    corr = float(np.corrcoef(truth_r, pred_r)[0, 1]) if np.std(pred_r) > 1e-8 else 0.
    return {"accuracy": float(np.mean(pred_c == truth_c)),
            "macro_f1": float(np.mean(f1)),
            "mae": float(np.mean(np.abs(pred_r-truth_r))),
            "pearson": corr}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             scenario: tuple[str, float] | None = None,
             start_fraction: float = .35) -> dict[str, float]:
    model.eval()
    true_c, pred_c, true_r, pred_r = [], [], [], []
    for batch in loader:
        if scenario is not None:
            batch = missing_block(batch, modality=scenario[0], fraction=scenario[1],
                                  start_fraction=start_fraction)
        batch = to_device(batch, device)
        logits, intensity = model(batch)
        true_c.append(batch["class_label"].cpu().numpy())
        pred_c.append(logits.argmax(1).cpu().numpy())
        true_r.append(batch["reg_label"].cpu().numpy())
        pred_r.append(intensity.cpu().numpy())
    return scores(*(np.concatenate(parts) for parts in (true_c, pred_c, true_r, pred_r)))


@torch.inference_mode()
def evaluate_unimodal(model: UnalignedFusion, loader: DataLoader,
                      device: torch.device) -> dict[str, dict | None]:
    """Measure each head on samples where its own modality is observed."""
    model.eval()
    collected = {name: [[], [], [], []] for name in MODALITIES}
    for batch in loader:
        batch = to_device(batch, device)
        _, _, aux = model(batch, return_aux=True)
        for index, name in enumerate(MODALITIES):
            mask = aux["availability"][:, index]
            if not mask.any():
                continue
            parts = (batch["class_label"][mask],
                     aux["logits"][mask, index].argmax(dim=1),
                     batch["reg_label"][mask],
                     aux["intensity"][mask, index])
            for target, part in zip(collected[name], parts):
                target.append(part.cpu().numpy())
    result = {}
    for name, parts in collected.items():
        if not parts[0]:
            result[name] = None
            continue
        arrays = [np.concatenate(part) for part in parts]
        result[name] = {"n": len(arrays[0]), "metrics": scores(*arrays)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--text-cache", "--raw-text-cache", dest="text_cache", type=Path,
                        help="Use a BERT feature cache generated from text_bert or raw_text")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "unaligned_aux")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--aux-weight", type=float, default=.2,
                        help="Weight of unimodal classification/regression loss; 0 restores the old model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--strong-robust", action="store_true",
                        help="Include full text dropout to learn audio/vision fallback")
    parser.add_argument("--validation-only", action="store_true",
                        help="Train/select on validation and leave the test split untouched")
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Only for checking that the training code runs; results are not valid experiments")
    args = parser.parse_args()
    if args.no_robust and args.strong_robust:
        parser.error("--no-robust and --strong-robust cannot be combined")
    if args.aux_weight < 0:
        parser.error("--aux-weight must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, scales = load_data(args.source, args.text_cache)
    training = datasets["train"]
    if args.smoke_batches:
        training = Subset(training, range(min(len(training), args.smoke_batches*args.batch_size)))
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(datasets["valid"], batch_size=args.batch_size*2, num_workers=0)
    test_loader = DataLoader(datasets["test"], batch_size=args.batch_size*2, num_workers=0)
    model = UnalignedFusion(auxiliary_heads=args.aux_weight > 0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    labels = datasets["train"].fields["class_label"].numpy()
    counts = np.bincount(labels, minlength=3)
    weights = np.sqrt(counts.sum()/(3*np.maximum(counts, 1)))
    weights /= weights.mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    ce_each = nn.CrossEntropyLoss(weight=ce.weight, reduction="none")
    reg = nn.SmoothL1Loss(beta=.5)
    reg_each = nn.SmoothL1Loss(beta=.5, reduction="none")
    best_score, best_epoch, no_improvement = -1e9, 0, 0
    history = []
    checkpoint_path = args.output / "best.pt"
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.
        total_aux_loss = 0.
        for batch in train_loader:
            if args.strong_robust:
                draw = random.random()
                if draw < .3:
                    batch = missing_block(batch, modality="text", fraction=1.)
                elif draw < .85:
                    batch = missing_block(batch)
            elif not args.no_robust and random.random() < .7:
                batch = missing_block(batch)
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            if args.aux_weight:
                logits, intensity, aux = model(batch, return_aux=True)
            else:
                logits, intensity = model(batch)
            loss = ce(logits, batch["class_label"]) + .5*reg(intensity, batch["reg_label"])
            if args.aux_weight:
                aux_loss = unimodal_loss(aux, batch, ce_each, reg_each)
                loss = loss + args.aux_weight * aux_loss
                total_aux_loss += float(aux_loss.detach())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += float(loss.detach())
        clean = evaluate(model, valid_loader, device)
        unimodal_clean = evaluate_unimodal(model, valid_loader, device) if args.aux_weight else None
        missing = {name:evaluate(model, valid_loader, device, (name, .5))
                   for name in MODALITIES}
        text_fully_missing = evaluate(model, valid_loader, device, ("text", 1.)) if args.strong_robust else None
        robust_f1 = float(np.mean([m["macro_f1"] for m in missing.values()]))
        robust_mae = float(np.mean([m["mae"] for m in missing.values()]))
        if args.strong_robust:
            selection = (.4*clean["macro_f1"] + .4*robust_f1
                         + .2*text_fully_missing["macro_f1"]
                         - .1*(clean["mae"]+robust_mae+text_fully_missing["mae"])/3)
        else:
            selection = .5*clean["macro_f1"] + .5*robust_f1 - .1*(clean["mae"]+robust_mae)/2
        record = {"epoch":epoch, "train_loss":total_loss/len(train_loader),
                  "train_aux_loss":total_aux_loss/len(train_loader) if args.aux_weight else None,
                  "valid_clean":clean, "valid_missing_half":missing,
                  "valid_unimodal_clean":unimodal_clean,
                  "valid_text_fully_missing":text_fully_missing,
                  "selection_score":selection}
        history.append(record)
        print(f"epoch {epoch:02d} loss={record['train_loss']:.4f} "
              f"val_F1={clean['macro_f1']:.3f} val_MAE={clean['mae']:.3f} "
              f"missing_F1={robust_f1:.3f}", flush=True)
        if selection > best_score + 1e-4:
            best_score, best_epoch, no_improvement = selection, epoch, 0
            torch.save({"state_dict":model.state_dict(), "scales":scales,
                        "config":{"width":128,"heads":4,"dropout":.15,
                                  "auxiliary_heads":args.aux_weight > 0},
                        "seed":args.seed,"epoch":epoch,"source":str(args.source),
                        "aux_weight":args.aux_weight,
                        "text_source":("frozen_bert_cache" if args.text_cache else "provided_text"),
                        "text_cache":str(args.text_cache) if args.text_cache else None},
                       checkpoint_path)
        else:
            no_improvement += 1
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if no_improvement >= 4 and not args.smoke_batches:
            break
    if args.smoke_batches:
        print("SMOKE RUN ONLY: skip test evaluation and do not use this checkpoint for submission")
        return
    if args.validation_only:
        print(f"VALIDATION ONLY: best epoch {best_epoch}; test split was not evaluated")
        return
    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    test_results = {"clean":evaluate(model, test_loader, device),
                    **{f"{name}_missing_{int(rate*100)}":evaluate(model, test_loader, device,
                         (name, rate)) for name in MODALITIES for rate in (.3, .6)}}
    result = {"best_epoch":best_epoch, "device":str(device),
              "validation_selection_score":best_score, "test":test_results,
              "test_unimodal_clean": (evaluate_unimodal(model, test_loader, device)
                                      if args.aux_weight else None)}
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
