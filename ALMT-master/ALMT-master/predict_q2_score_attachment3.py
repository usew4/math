"""Predict sentiment scores and ±0.2 classes for Attachment-3 aligned samples."""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import torch

from train_q2_aligned import AMP_DTYPE, LABELS, ROOT, build_q2_model
from train_q2_score import score_to_class

INPUT = Path(
    r"D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据\E题数据"
) / "附件3-模态缺失特征样本" / "对齐版本"


def main() -> None:
    default_output = ROOT / "runs" / "q2_almt_score_threshold02"
    if not (default_output / "best.pt").is_file():
        default_output = ROOT / "runs" / "q2_almt_score_threshold05"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=default_output / "best.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.2)
    cli = parser.parse_args()
    if not 0 < cli.threshold < 3:
        parser.error("--threshold must be in (0, 3)")
    checkpoint = cli.checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train the score model first: {checkpoint}")
    paths = sorted(INPUT.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 aligned Attachment-3 samples, found {len(paths)}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = saved["config"]
    threshold = cli.threshold
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(config["base"]["amp"]) and device.type == "cuda"
    model = build_q2_model(config, device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    ids, texts, audios, visions = [], [], [], []
    for path in paths:
        with path.open("rb") as stream:
            item = pickle.load(stream)["test"]
        if not {"text_bert", "audio", "vision"}.issubset(item):
            raise ValueError(f"Unexpected fields in {path.name}: {list(item)}")
        ids.append(path.stem)
        texts.append(np.asarray(item["text_bert"][0], dtype=np.int64))
        audios.append(np.nan_to_num(np.asarray(item["audio"][0], dtype=np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0))
        visions.append(np.nan_to_num(np.asarray(item["vision"][0], dtype=np.float32),
                                     nan=0.0, posinf=0.0, neginf=0.0))
    text = np.stack(texts)
    audio = np.stack(audios)
    vision = np.stack(visions)
    if text.shape != (30, 3, 50) or audio.shape != (30, 50, 74) or vision.shape != (30, 50, 35):
        raise ValueError(f"Unexpected Attachment-3 shapes: {text.shape}, {audio.shape}, {vision.shape}")
    scores = []
    with torch.inference_mode():
        for start in range(0, len(ids), int(config["base"]["batch_size"])):
            end = start + int(config["base"]["batch_size"])
            with torch.autocast(device_type=device.type, dtype=AMP_DTYPE, enabled=amp):
                output_score = model(
                    torch.from_numpy(vision[start:end]).to(device),
                    torch.from_numpy(audio[start:end]).to(device),
                    torch.from_numpy(text[start:end]).to(device))
            scores.extend(output_score.squeeze(-1).float().clamp(-3, 3).cpu().tolist())
    scores = np.asarray(scores, dtype=np.float32)
    classes = score_to_class(scores, threshold)
    tag = f"threshold{threshold:g}".replace(".", "p")
    result_file = (cli.output if cli.output is not None else
                   default_output / f"attachment3_score_predictions_{tag}.csv")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with result_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "pred_score", "class_id", "class_name", "neutral_threshold"])
        for sample_id, score, cls in zip(ids, scores, classes):
            writer.writerow([sample_id, round(float(score), 6), int(cls),
                             LABELS[int(cls)], threshold])
    print(f"saved {len(ids)} score-and-class predictions to {result_file}", flush=True)


if __name__ == "__main__":
    main()
