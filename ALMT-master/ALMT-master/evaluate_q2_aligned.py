"""Evaluate the selected ALMT checkpoint on the held-out Attachment-2 test split once."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from train_q2_aligned import ROOT, build_q2_model, evaluate, load_splits


def main() -> None:
    output = ROOT / "runs" / "q2_almt_aligned"
    checkpoint = output / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train first; missing checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    model = build_q2_model(config, device)
    model.load_state_dict(saved["state_dict"])
    test = load_splits(Path(saved["source"]))["test"]
    loader = DataLoader(test, batch_size=int(config["base"]["batch_size"]),
                        shuffle=False, num_workers=0)
    result = {"checkpoint_epoch": saved["epoch"], "test": evaluate(model, loader, device, amp)}
    (output / "test_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
