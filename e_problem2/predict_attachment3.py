"""Infer the 30 unaligned missing-modality samples using the trained Q2 model."""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from data_unaligned import prepare_unlabeled
from model_unaligned import UnalignedFusion
from paths import (BASE_SITE_PACKAGES, BERT_MODEL_DIR, BERT_VENDOR_DIR,
                   MISSING_UNALIGNED_DIR, check_source_paths)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = MISSING_UNALIGNED_DIR
LABELS = ("Negative", "Neutral", "Positive")


def load_bert():
    # torch must be imported from the selected PyTorch environment first.
    os.environ.setdefault("USE_TF", "0")
    sys.path.append(str(BASE_SITE_PACKAGES))
    sys.path.insert(0, str(BERT_VENDOR_DIR))
    from transformers import AutoModel, AutoTokenizer
    model_dir = BERT_MODEL_DIR
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    bert = AutoModel.from_pretrained(str(model_dir))
    bert.eval()
    return tokenizer, bert


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "unaligned_robust" / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "unaligned_robust" / "attachment3_predictions.csv")
    args = parser.parse_args()
    check_source_paths(need_bert=True, need_missing=True)
    paths = sorted(args.input.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 unaligned samples, found {len(paths)}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fusion = UnalignedFusion(**saved["config"])
    fusion.load_state_dict(saved["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion.to(device).eval()
    texts, audios, visions, ids = [], [], [], []
    for path in paths:
        with path.open("rb") as stream:
            entry = pickle.load(stream)["test"]
        if set(entry) != {"raw_text", "audio", "vision"}:
            raise ValueError(f"Unexpected fields in {path.name}: {list(entry)}")
        ids.append(path.stem)
        texts.append(str(entry["raw_text"][0]))
        audios.append(np.asarray(entry["audio"][0], dtype=np.float32))
        visions.append(np.asarray(entry["vision"][0], dtype=np.float32))
    tokenizer, bert = load_bert()
    bert.to(device)
    encoded = tokenizer(texts, max_length=50, truncation=True,
                        padding="max_length", return_tensors="pt")
    with torch.inference_mode():
        vectors = []
        for start in range(0, len(texts), 8):
            part = {key:value[start:start+8].to(device) for key,value in encoded.items()}
            vectors.append(bert(**part).last_hidden_state.cpu().numpy())
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    batch = prepare_unlabeled(np.concatenate(vectors), np.stack(audios), np.stack(visions),
                              encoded["attention_mask"].numpy().astype(bool), saved["scales"])
    batch = {key:value.to(device) for key,value in batch.items()}
    with torch.inference_mode():
        logits, intensity = fusion(batch)
        probabilities = logits.softmax(dim=1).cpu().numpy()
        predicted = probabilities.argmax(axis=1)
        intensity = intensity.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class_id", "annotation", "regression_label",
                         "prob_negative", "prob_neutral", "prob_positive"])
        for i, name in enumerate(ids):
            writer.writerow([name, int(predicted[i]), LABELS[predicted[i]],
                             round(float(intensity[i]), 6),
                             *[round(float(p), 6) for p in probabilities[i]]])
    print(f"saved {len(ids)} predictions to {args.output}")


if __name__ == "__main__":
    main()
