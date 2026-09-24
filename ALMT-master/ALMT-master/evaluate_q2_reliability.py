"""Evaluate the selected mask-aware Q2 model on the held-out test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_q2_aligned import ROOT
from train_q2_reliability import build_model, evaluate, write_rows
from train_q2_score import load_score_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "runs" / "q2_text_anchor_reliability" / "best.pt")
    cli = parser.parse_args()
    if not cli.checkpoint.is_file():
        raise FileNotFoundError(f"Train and select the model first: {cli.checkpoint}")
    saved = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    datasets = load_score_splits(Path(saved["source"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    model = build_model(config, datasets["train"], device)
    model.load_state_dict(saved["state_dict"])
    test = DataLoader(datasets["test"], batch_size=int(config["base"]["batch_size"]),
                      shuffle=False, num_workers=0)
    result, rows = evaluate(model, test, device, amp,
                            float(config["task"]["neutral_threshold"]))
    output = cli.checkpoint.parent
    write_rows(output / "test_predictions.csv", rows)
    payload = {"checkpoint": str(cli.checkpoint), "checkpoint_epoch": saved["epoch"],
               "test": result}
    (output / "test_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
