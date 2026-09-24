"""Reclassify saved ALMT validation scores using ±0.2 without retraining."""
from __future__ import annotations

import csv
import json

import numpy as np

from train_q2_aligned import ROOT
from train_q2_score import score_metrics, score_to_class, write_rows


def main() -> None:
    output = ROOT / "runs" / "q2_almt_score_threshold05"
    source = output / "valid_predictions.csv"
    if not source.is_file():
        raise FileNotFoundError(f"First train the score model: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    threshold = 0.2
    true_score = np.asarray([float(row["true_score"]) for row in rows], dtype=np.float32)
    pred_score = np.asarray([float(row["pred_score"]) for row in rows], dtype=np.float32)
    true_class = np.asarray([int(row["true_class"]) for row in rows], dtype=np.int64)
    pred_class = score_to_class(pred_score, threshold)
    for row, label in zip(rows, pred_class):
        row["pred_class"] = int(label)
    result = score_metrics(true_score, pred_score, true_class, threshold)
    write_rows(output / "valid_predictions_threshold0p2.csv", rows)
    (output / "validation_threshold0p2.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"±{threshold}: val_acc={result['accuracy']:.2%} "
          f"val_macro_F1={result['macro_f1']:.3f} val_MAE={result['mae']:.3f}", flush=True)
    print(f"Saved: {output / 'valid_predictions_threshold0p2.csv'}", flush=True)


if __name__ == "__main__":
    main()
