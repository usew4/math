"""Evaluate the selected ALMT score model on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_q2_aligned import ROOT, build_q2_model
from train_q2_score import evaluate_score, load_score_splits, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=(ROOT / "runs" / "q2_almt_score_threshold02" / "best.pt"
                                 if (ROOT / "runs" / "q2_almt_score_threshold02" / "best.pt").is_file()
                                 else ROOT / "runs" / "q2_almt_score_threshold05" / "best.pt"))
    parser.add_argument("--threshold", type=float, default=0.2)
    cli = parser.parse_args()
    if not 0 < cli.threshold < 3:
        parser.error("--threshold must be in (0, 3)")
    checkpoint = cli.checkpoint
    output = checkpoint.parent
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train the score model first: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    threshold = cli.threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    model = build_q2_model(config, device)
    model.load_state_dict(saved["state_dict"])
    test = load_score_splits(Path(saved["source"]))["test"]
    loader = DataLoader(test, batch_size=int(config["base"]["batch_size"]),
                        shuffle=False, num_workers=0)
    metrics, rows = evaluate_score(model, loader, device, amp, threshold)
    suffix = f"threshold{threshold:g}".replace(".", "p")
    write_rows(output / f"test_predictions_{suffix}.csv", rows)
    result = {"checkpoint_epoch": saved["epoch"], "test": metrics}
    (output / f"test_result_{suffix}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
