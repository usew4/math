"""Predict Attachment 3 with the DecAlign Q2 checkpoint."""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))

from data_unaligned import prepare_unlabeled  # noqa: E402
from paths import MISSING_UNALIGNED_DIR, check_source_paths  # noqa: E402
from predict_attachment3 import LABELS, load_bert  # noqa: E402
from q2_model import Q2DecAlign  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MISSING_UNALIGNED_DIR)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "runs" / "decalign_q2_unaligned" / "best.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "runs" / "decalign_q2_unaligned" / "attachment3_predictions.csv")
    args = parser.parse_args()
    check_source_paths(need_bert=True, need_missing=True)
    paths = sorted(args.input.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 unaligned samples, found {len(paths)}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("text_input") != "pkl.text" or saved.get("feature_mode", "unaligned") != "unaligned":
        raise ValueError("Checkpoint must be trained on unaligned_50.pkl with `text` features")
    model = Q2DecAlign(SimpleNamespace(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ids, texts, audios, visions = [], [], [], []
    for path in paths:
        with path.open("rb") as stream:
            item = pickle.load(stream)["test"]
        if set(item) != {"raw_text", "audio", "vision"}:
            raise ValueError(f"Unexpected fields in {path.name}: {list(item)}")
        ids.append(path.stem)
        texts.append(str(item["raw_text"][0]))
        audios.append(np.asarray(item["audio"][0], dtype=np.float32))
        visions.append(np.asarray(item["vision"][0], dtype=np.float32))
    # Attachment 3 does not contain `text`. Recreate the same 50 x 768 BERT
    # representation from raw_text, as verified against Attachment 2.
    tokenizer, bert = load_bert()
    bert.to(device)
    tokens = tokenizer(texts, max_length=50, truncation=True,
                       padding="max_length", return_tensors="pt")
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), 8):
            part = {k: v[start:start + 8].to(device) for k, v in tokens.items()}
            vectors.append(bert(**part).last_hidden_state.cpu().numpy())
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    batch = prepare_unlabeled(
        np.concatenate(vectors), np.stack(audios), np.stack(visions),
        tokens["attention_mask"].numpy().astype(bool), saved["scales"])
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        outputs = model(batch)
        probs = outputs["output_logit"].softmax(-1).cpu().numpy()
        intensity = outputs["intensity"].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class_id", "annotation", "regression_label",
                         "prob_negative", "prob_neutral", "prob_positive"])
        for i, sample_id in enumerate(ids):
            class_id = int(probs[i].argmax())
            writer.writerow([sample_id, class_id, LABELS[class_id],
                             round(float(intensity[i]), 6),
                             *[round(float(value), 6) for value in probs[i]]])
    print(f"saved {len(ids)} predictions to {args.output}")


if __name__ == "__main__":
    main()
