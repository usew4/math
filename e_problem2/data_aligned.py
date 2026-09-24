"""Load Attachment-2 word-aligned features for the DecAlign Q2 experiment."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from data_unaligned import FeatureDataset, apply_scale, fit_scale
from paths import ALIGNED_PKL

DEFAULT_SOURCE = ALIGNED_PKL


def load_data(path: Path = DEFAULT_SOURCE):
    with path.open("rb") as stream:
        source = pickle.load(stream)
    if set(source) != {"train", "valid", "test"}:
        raise ValueError("Expected train/valid/test splits")
    prepared = {}
    for split, entry in source.items():
        text_mask = np.asarray(entry["text_bert"][:, 1, :], dtype=bool)
        fields = {"text_mask": text_mask,
                  "class_label": np.asarray(entry["classification_labels"], dtype=np.int64),
                  "reg_label": np.asarray(entry["regression_labels"], dtype=np.float32)}
        for name, shape in (("text", (50, 768)), ("audio", (50, 74)),
                            ("vision", (50, 35))):
            values = np.asarray(entry[name], dtype=np.float32).copy()
            if values.shape[1:] != shape:
                raise ValueError(f"Unexpected {split} {name} shape: {values.shape}")
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if name == "text":
                observed = text_mask
            else:
                observed = np.any(values != 0, axis=-1) & text_mask
                fields[f"{name}_mask"] = observed
            values[~observed] = 0
            fields[name] = values
        if fields["text"].shape[0] != len(entry["id"]):
            raise ValueError(f"Mismatched sample count in {split}")
        prepared[split] = fields
    if any(set(source[a]["id"]) & set(source[b]["id"])
           for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Overlapping sample IDs between splits")
    scales = {name: fit_scale(prepared["train"][name],
                              prepared["train"][f"{name}_mask"])
              for name in ("audio", "vision")}
    for fields in prepared.values():
        for name in ("audio", "vision"):
            fields[name] = apply_scale(fields[name], fields[f"{name}_mask"], scales[name])
    return {name: FeatureDataset(fields) for name, fields in prepared.items()}, scales


def prepare_unlabeled(text, audio, vision, text_mask, scales):
    """Prepare Attachment-3 aligned tensors after its token IDs are BERT-encoded."""
    text = np.asarray(text, np.float32).copy()
    text_mask = np.asarray(text_mask, bool)
    if text.shape[1:] != (50, 768) or audio.shape[1:] != (50, 74) or vision.shape[1:] != (50, 35):
        raise ValueError("Expected aligned 50-step text/audio/vision features")
    text[~text_mask] = 0
    fields = {"text": text, "text_mask": text_mask}
    for name, array in (("audio", audio), ("vision", vision)):
        values = np.nan_to_num(np.asarray(array, np.float32), nan=0.0,
                               posinf=0.0, neginf=0.0)
        mask = np.any(values != 0, axis=-1) & text_mask
        fields[name] = apply_scale(values, mask, scales[name])
        fields[f"{name}_mask"] = mask
    return {key: torch.from_numpy(value) for key, value in fields.items()}
