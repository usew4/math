"""Evaluate a selected two-head ALMT checkpoint on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_q2_aligned import ROOT
from train_q2_multitask import build_model, evaluate, write_rows
from train_q2_score import load_score_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs" / "q2_almt_multitask" / "best_score.pt")
    cli = parser.parse_args()
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(f"Train the multi-task model first: {cli.checkpoint}")
    saved = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    threshold = float(config["task"]["neutral_threshold"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    model = build_model(config, device)
    model.load_state_dict(saved["state_dict"])
    test = load_score_splits(Path(saved["source"]))["test"]
    loader = DataLoader(test, batch_size=int(config["base"]["batch_size"]),
                        shuffle=False, num_workers=0)
    result, rows = evaluate(model, loader, device, amp, threshold)
    path = cli.checkpoint.parent
    suffix = cli.checkpoint.stem
    write_rows(path / f"test_predictions_{suffix}.csv", rows)
    payload = {"checkpoint": str(cli.checkpoint), "checkpoint_epoch": saved["epoch"],
               "test": result}
    (path / f"test_result_{suffix}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
