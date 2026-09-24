"""Predict Attachment-3 aligned samples with a trained aligned DecAlign checkpoint."""
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

from data_aligned import prepare_unlabeled  # noqa: E402
from paths import MISSING_ALIGNED_DIR, check_source_paths  # noqa: E402
from predict_attachment3 import LABELS, load_bert  # noqa: E402
from q2_model import Q2DecAlign  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MISSING_ALIGNED_DIR)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "runs" / "decalign_q2_aligned_final" / "best.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "runs" / "decalign_q2_aligned_final" / "attachment3_predictions.csv")
    args = parser.parse_args()
    check_source_paths(need_bert=True)
    paths = sorted(args.input.glob("*.pkl"))
    if len(paths) != 30:
        raise ValueError(f"Expected 30 aligned Attachment-3 samples, found {len(paths)}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("feature_mode") != "aligned":
        raise ValueError("The checkpoint must be trained on aligned_50.pkl")
    model = Q2DecAlign(SimpleNamespace(**saved["config"]))
    model.load_state_dict(saved["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ids, tokens, audios, visions = [], [], [], []
    for path in paths:
        with path.open("rb") as stream:
            item = pickle.load(stream)["test"]
        if set(item) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"Unexpected fields in {path.name}: {list(item)}")
        ids.append(path.stem)
        tokens.append(np.asarray(item["text_bert"][0], dtype=np.int64))
        audios.append(np.asarray(item["audio"][0], dtype=np.float32))
        visions.append(np.asarray(item["vision"][0], dtype=np.float32))
    encoded = np.stack(tokens)
    if encoded.shape != (30, 3, 50):
        raise ValueError(f"Unexpected text_bert shape: {encoded.shape}")
    _, bert = load_bert()
    bert.to(device)
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(encoded), 8):
            part = torch.from_numpy(encoded[start:start + 8]).to(device)
            vectors.append(bert(input_ids=part[:, 0], attention_mask=part[:, 1],
                                token_type_ids=part[:, 2]).last_hidden_state.cpu().numpy())
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()
    batch = prepare_unlabeled(np.concatenate(vectors), np.stack(audios), np.stack(visions),
                              encoded[:, 1, :].astype(bool), saved["scales"])
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        result = model(batch)
        probabilities = result["output_logit"].softmax(-1).cpu().numpy()
        intensity = result["intensity"].cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class_id", "annotation", "regression_label",
                         "prob_negative", "prob_neutral", "prob_positive"])
        for i, sample_id in enumerate(ids):
            class_id = int(probabilities[i].argmax())
            writer.writerow([sample_id, class_id, LABELS[class_id],
                             round(float(intensity[i]), 6),
                             *[round(float(x), 6) for x in probabilities[i]]])
    print(f"saved {len(ids)} aligned predictions to {args.output}")


if __name__ == "__main__":
    main()
