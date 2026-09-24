"""Compare clean and missing-block training by type, duration and position."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_unaligned import DEFAULT_SOURCE, load_data
from model_unaligned import UnalignedFusion
from train_unaligned import MODALITIES, evaluate


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--text-cache", "--raw-text-cache", dest="text_cache", type=Path,
                        help="Use BERT features generated from text_bert or raw_text")
    parser.add_argument("--robust", type=Path, default=ROOT / "runs" / "unaligned_robust" / "best.pt")
    parser.add_argument("--clean", type=Path, default=ROOT / "runs" / "unaligned_clean" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "robustness_grid.csv")
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, _ = load_data(args.source, args.text_cache)
    loader = DataLoader(datasets["test"], batch_size=64, num_workers=0)
    rows = []
    for name, path in (("robust", args.robust), ("clean_training", args.clean)):
        saved = torch.load(path, map_location=device, weights_only=False)
        model = UnalignedFusion(**saved["config"]).to(device)
        model.load_state_dict(saved["state_dict"])
        scenarios = [("none", 0., -1.)] + [
            (modality, rate, pos) for modality in MODALITIES
            for rate in (.3, .6) for pos in (0., .5, 1.)]
        for modality, rate, pos in scenarios:
            metric = (evaluate(model, loader, device) if modality == "none" else
                      evaluate(model, loader, device, (modality, rate), start_fraction=pos))
            row = {"training":name, "missing_modality":modality,
                   "missing_fraction":rate, "position":pos, **metric}
            rows.append(row)
            print(f"{name} {modality} rate={rate:.1f} pos={pos:.1f} "
                  f"F1={metric['macro_f1']:.3f} MAE={metric['mae']:.3f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)


if __name__ == "__main__":
    main()
