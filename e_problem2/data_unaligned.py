"""Load the supplied unaligned CMU-MOSEI features without changing the splits."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from paths import UNALIGNED_PKL

DEFAULT_SOURCE = UNALIGNED_PKL


def pool_five(array: np.ndarray, lengths: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Reduce 500 consecutive positions to 100 without averaging missing zeros."""
    array = np.asarray(array)
    assert array.ndim == 3 and array.shape[1] == 500
    observed = np.any(array != 0, axis=2)
    if lengths is not None:
        observed &= np.arange(500)[None, :] < np.asarray(lengths)[:, None]
    blocks = array.astype(np.float32).reshape(len(array), 100, 5, array.shape[2])
    mask = observed.reshape(len(array), 100, 5)
    counts = mask.sum(axis=2)
    pooled = (blocks * mask[..., None]).sum(axis=2) / np.maximum(counts[..., None], 1)
    pooled[counts == 0] = 0
    return pooled.astype(np.float32), (counts > 0)


def fit_scale(features: np.ndarray, observed: np.ndarray) -> dict[str, np.ndarray]:
    """Fit channel statistics from training observations only."""
    values = features[observed].astype(np.float64)
    mean = values.mean(axis=0)
    std = np.maximum(values.std(axis=0), 1e-4)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_scale(features: np.ndarray, observed: np.ndarray,
                stats: dict[str, np.ndarray]) -> np.ndarray:
    out = np.clip((features - stats["mean"]) / stats["std"], -10, 10).astype(np.float32)
    out[~observed] = 0
    return out


class FeatureDataset(Dataset):
    def __init__(self, fields: dict):
        self.fields = {key: torch.from_numpy(value) for key, value in fields.items()
                       if isinstance(value, np.ndarray)}

    def __len__(self) -> int:
        return len(self.fields["text"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.fields.items()}


def load_data(path: Path = DEFAULT_SOURCE,
              raw_text_cache: Path | None = None) -> tuple[dict[str, FeatureDataset], dict]:
    # The competition attachment is a Python pickle; load only the trusted local file.
    with path.open("rb") as stream:
        source = pickle.load(stream)
    assert set(source) == {"train", "valid", "test"}
    cache = None
    if raw_text_cache is not None:
        with np.load(raw_text_cache, allow_pickle=False) as stored:
            cache = {split: {
                "text": stored[f"{split}_text"].astype(np.float32),
                "mask": stored[f"{split}_mask"].astype(bool),
            } for split in ("train", "valid", "test")}
    prepared = {}
    for split, entry in source.items():
        text_mask = (cache[split]["mask"] if cache is not None else
                     np.asarray(entry["text_bert"][:, 1, :], dtype=bool))
        audio, audio_mask = pool_five(entry["audio"], np.asarray(entry["audio_lengths"]))
        vision, vision_mask = pool_five(entry["vision"], np.asarray(entry["vision_lengths"]))
        fields = {
            "text": (cache[split]["text"] if cache is not None else
                     np.asarray(entry["text"], dtype=np.float32)),
            "audio": audio,
            "vision": vision,
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "vision_mask": vision_mask,
            "class_label": np.asarray(entry["classification_labels"], dtype=np.int64),
            "reg_label": np.asarray(entry["regression_labels"], dtype=np.float32),
        }
        assert fields["text"].shape[1:] == (50, 768)
        assert fields["text"].shape[0] == len(entry["id"]), split
        assert fields["audio"].shape[1:] == (100, 74)
        assert fields["vision"].shape[1:] == (100, 35)
        prepared[split] = fields
    assert all(not (set(source[a]["id"]) & set(source[b]["id"]))
               for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")))
    scales = {
        "audio": fit_scale(prepared["train"]["audio"], prepared["train"]["audio_mask"]),
        "vision": fit_scale(prepared["train"]["vision"], prepared["train"]["vision_mask"]),
    }
    for fields in prepared.values():
        for name in ("audio", "vision"):
            fields[name] = apply_scale(fields[name], fields[f"{name}_mask"], scales[name])
        # The supplied BERT tensor has nonzero values at padding positions.
        fields["text"][~fields["text_mask"]] = 0
    return {name: FeatureDataset(fields) for name, fields in prepared.items()}, scales


def prepare_unlabeled(text_embedding: np.ndarray, audio: np.ndarray,
                      vision: np.ndarray, text_mask: np.ndarray,
                      scales: dict) -> dict[str, torch.Tensor]:
    """Prepare one or more Attachment 3 samples using the same input interface."""
    audio, audio_mask = pool_five(audio)
    vision, vision_mask = pool_five(vision)
    text_embedding = np.asarray(text_embedding, np.float32).copy()
    text_mask = np.asarray(text_mask, bool)
    text_embedding[~text_mask] = 0
    fields = {
        "text": text_embedding,
        "audio": apply_scale(audio, audio_mask, scales["audio"]),
        "vision": apply_scale(vision, vision_mask, scales["vision"]),
        "text_mask": text_mask,
        "audio_mask": audio_mask,
        "vision_mask": vision_mask,
    }
    return {name: torch.from_numpy(value) for name, value in fields.items()}
