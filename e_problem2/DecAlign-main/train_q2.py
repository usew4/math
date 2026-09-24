"""Train the DecAlign source on E-question-2 aligned or unaligned features."""
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
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from data_aligned import DEFAULT_SOURCE as ALIGNED_SOURCE, load_data as load_aligned  # noqa: E402
from data_unaligned import DEFAULT_SOURCE as UNALIGNED_SOURCE, load_data as load_unaligned  # noqa: E402
from train_unaligned import scores  # noqa: E402
from config import get_config_regression  # noqa: E402
from q2_model import Q2DecAlign  # noqa: E402

MODALITIES = ("text", "audio", "vision")


def model_config(args):
    cfg = get_config_regression("decalign", "mosei")
    cfg.use_bert = False  # Attachment-2 `text` is already N x 50 x 768.
    cfg.need_data_aligned = args.feature_mode == "aligned"
    cfg.train_mode = "classification"
    cfg.head_type = args.head_type
    cfg.num_classes = 3
    cfg.feature_dims = [768, 74, 35]
    cfg.dst_feature_dim_nheads = [args.hidden, args.num_heads]
    cfg.nlevels = args.depth
    kernel = 5 if args.feature_mode == "aligned" else 1
    cfg.conv1d_kernel_size_l = kernel
    cfg.conv1d_kernel_size_a = kernel
    cfg.conv1d_kernel_size_v = kernel
    cfg.conv_same_padding = args.feature_mode == "aligned"
    cfg.num_prototypes = args.num_prototypes
    cfg.gmm_em_iters = args.gmm_iters
    cfg.ot_num_iters = args.ot_iters
    cfg.use_sequence_mask = True
    cfg.use_temporal_position = args.feature_mode == "unaligned"
    cfg.temporal_position_scale = args.temporal_position_scale
    cfg.attn_mask = False  # Full utterance classification, not causal prediction.
    cfg.pde_dim = args.hidden
    cfg.alpha = args.alpha
    cfg.beta = args.beta
    cfg.dec_loss_weight = args.dec_weight
    cfg.text_dropout = args.dropout
    cfg.output_dropout = args.dropout
    cfg.embed_dropout = args.dropout
    return cfg


def hide_block(batch, *, modality=None, fraction=None, position=None):
    """Mask a contiguous observed interval, keeping each modality's own time axis."""
    result = {key: value.clone() for key, value in batch.items()}
    for i in range(len(result["text"])):
        choices = [name for name in MODALITIES if result[f"{name}_mask"][i].any()]
        if not choices or (modality is not None and modality not in choices):
            continue
        chosen = modality or random.choice(choices)
        observed = torch.nonzero(result[f"{chosen}_mask"][i]).flatten()
        length = int(observed[-1]) + 1
        rate = fraction if fraction is not None else random.uniform(.2, .6)
        width = min(length, max(1, round(length * rate)))
        start = (round((length - width) * position) if position is not None
                 else random.randint(0, length - width))
        result[chosen][i, start:start + width] = 0
        result[f"{chosen}_mask"][i, start:start + width] = False
    return result


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def loss_fn(outputs, batch, args):
    classification = F.cross_entropy(outputs["output_logit"], batch["class_label"],
                                     weight=args.class_weights)
    regression = F.smooth_l1_loss(outputs["intensity"], batch["reg_label"], beta=.5)
    total = (classification + args.reg_weight * regression
             + args.dec_weight * outputs["dec_loss"]
             + args.alpha * outputs["hete_loss"]
             + args.beta * outputs["homo_loss"])
    return total


def train_epoch(model, loader, optimizer, args, device):
    model.train()
    losses = []
    for i, batch in enumerate(loader):
        if args.smoke_batches and i >= args.smoke_batches:
            break
        if not args.no_robust and random.random() < args.robust_prob:
            batch = hide_block(batch)
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = loss_fn(outputs, batch, args)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at batch {i}")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.inference_mode()
def evaluate(model, loader, device, *, scenario=None, position=.35, limit=0):
    model.eval()
    true_c, pred_c, true_r, pred_r = [], [], [], []
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        if scenario is not None:
            batch = hide_block(batch, modality=scenario[0], fraction=scenario[1],
                               position=position)
        batch = move(batch, device)
        outputs = model(batch)
        true_c.append(batch["class_label"].cpu().numpy())
        pred_c.append(outputs["output_logit"].argmax(-1).cpu().numpy())
        true_r.append(batch["reg_label"].cpu().numpy())
        pred_r.append(outputs["intensity"].cpu().numpy())
    return scores(*(np.concatenate(items) for items in (true_c, pred_c, true_r, pred_r)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-mode", choices=("aligned", "unaligned"), default="aligned")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-prototypes", type=int, default=3)
    parser.add_argument("--gmm-iters", type=int, default=3)
    parser.add_argument("--ot-iters", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--temporal-position-scale", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--reg-weight", type=float, default=.5)
    parser.add_argument("--dec-weight", type=float, default=.1)
    parser.add_argument("--alpha", type=float, default=.005)
    parser.add_argument("--beta", type=float, default=.005)
    parser.add_argument("--robust-prob", type=float, default=.7)
    parser.add_argument("--class-weight", choices=("none", "sqrt_balanced", "balanced"),
                        default="none")
    parser.add_argument("--head-type", choices=("flat", "hierarchical"), default="flat")
    parser.add_argument("--no-robust", action="store_true")
    parser.add_argument("--clean-only", action="store_true",
                        help="No simulated missingness in training, validation or test grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()
    if args.clean_only:
        args.no_robust = True
        args.skip_grid = True
    if args.source is None:
        args.source = ALIGNED_SOURCE if args.feature_mode == "aligned" else UNALIGNED_SOURCE
    if args.output is None:
        run_name = ("decalign_q2_aligned_final" if args.feature_mode == "aligned"
                    else "decalign_q2_unaligned")
        args.output = PROJECT / "runs" / run_name
    if args.hidden % args.num_heads:
        parser.error("--hidden must be divisible by --num-heads")
    if not args.source.is_file():
        parser.error(f"Missing source: {args.source}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = load_aligned if args.feature_mode == "aligned" else load_unaligned
    datasets, scales = loader(args.source)  # Uses pkl `text`, no BERT cache.
    class_counts = np.bincount(datasets["train"].fields["class_label"].numpy(), minlength=3)
    if args.class_weight == "none":
        args.class_weights = None
    else:
        weights = class_counts.sum() / (3 * np.maximum(class_counts, 1))
        if args.class_weight == "sqrt_balanced":
            weights = np.sqrt(weights)
        weights = weights / np.average(weights, weights=class_counts)
        args.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    if args.smoke_batches:
        n = args.smoke_batches * args.batch_size
        datasets = {k: Subset(v, range(min(n, len(v)))) for k, v in datasets.items()}
    loaders = {k: DataLoader(v, batch_size=args.batch_size, shuffle=k == "train",
                             num_workers=0) for k, v in datasets.items()}
    cfg = model_config(args)
    model = Q2DecAlign(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    args.output.mkdir(parents=True, exist_ok=True)
    history, best_score, best_epoch, stale = [], -float("inf"), 0, 0
    steps = 50 if args.feature_mode == "aligned" else 100
    print(f"device={device} mode={args.feature_mode} output={args.output} "
          f"robust_prob={0 if args.no_robust else args.robust_prob} "
          f"class_weight={args.class_weight} head_type={args.head_type} "
          f"input=text(50,768),audio({steps},74),vision({steps},35) "
          f"train={len(datasets['train'])} valid={len(datasets['valid'])}", flush=True)
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, args, device)
        clean = evaluate(model, loaders["valid"], device, limit=args.smoke_batches)
        missing = ({} if args.clean_only else
                   {name: evaluate(model, loaders["valid"], device,
                                   scenario=(name, .5), limit=args.smoke_batches)
                    for name in MODALITIES})
        robust_f1 = (None if args.clean_only else
                     float(np.mean([part["macro_f1"] for part in missing.values()])))
        score = (clean["macro_f1"] if args.clean_only else
                 .5 * clean["macro_f1"] + .5 * robust_f1) - .05 * clean["mae"]
        record = {"epoch": epoch, "train_loss": loss, "valid_clean": clean,
                  "valid_missing_half": missing, "selection_score": score}
        history.append(record)
        (args.output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if score > best_score + 1e-4:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "scales": scales,
                        "config": vars(cfg), "epoch": epoch,
                        "source": str(args.source), "feature_mode": args.feature_mode,
                        "text_input": "pkl.text",
                        "training_config": {"class_weight": args.class_weight,
                                            "head_type": args.head_type,
                                            "clean_only": args.clean_only,
                                            "robust_prob": 0 if args.no_robust else args.robust_prob,
                                            "class_counts": class_counts.tolist()}},
                       args.output / "best.pt")
        else:
            stale += 1
        missing_display = f"{robust_f1:.3f}" if robust_f1 is not None else "skipped"
        print(f"epoch={epoch:02d} loss={loss:.4f} "
              f"val_acc={clean['accuracy']:.2%} val_F1={clean['macro_f1']:.3f} "
              f"missing_F1={missing_display}", flush=True)
        if stale >= args.patience and not args.smoke_batches:
            break
    if args.smoke_batches:
        print("SMOKE RUN ONLY; test split not evaluated", flush=True)
        return
    if args.validation_only:
        (args.output / "validation_complete.json").write_text(
            json.dumps({"best_epoch": best_epoch, "selection_score": best_score,
                        "feature_mode": args.feature_mode}, indent=2), encoding="utf-8")
        print(f"VALIDATION ONLY; best epoch={best_epoch}; test split not evaluated", flush=True)
        return
    state = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    result = {"best_epoch": best_epoch, "test_clean": evaluate(model, loaders["test"], device)}
    if not args.skip_grid:
        result["test_missing_grid"] = [
            {"modality": name, "fraction": rate, "position": pos,
             **evaluate(model, loaders["test"], device, scenario=(name, rate), position=pos)}
            for name in MODALITIES for rate in (.3, .6) for pos in (0., .5, 1.)
        ]
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
