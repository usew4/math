"""Predict the 30 unlabeled, unaligned Attachment 3 samples with adapted EBMC."""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT))

from data_unaligned import prepare_unlabeled  # noqa: E402
from paths import MISSING_UNALIGNED_DIR, check_source_paths  # noqa: E402
from predict_attachment3 import load_bert  # noqa: E402
from ebmc import build_model  # noqa: E402
from train_q2 import ebmc_inputs, make_args  # noqa: E402


LABELS = ("Negative", "Neutral", "Positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MISSING_UNALIGNED_DIR)
    parser.add_argument("--checkpoint", type=Path,
                        default=PROJECT / "runs" / "ebmc_q2" / "best.pt")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "runs" / "ebmc_q2" / "attachment3_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    check_source_paths(need_bert=True, need_missing=True)
    files = sorted(args.input.glob("*.pkl"))
    if len(files) != 30:
        raise ValueError(f"Expected 30 Attachment 3 files, found {len(files)}")
    if not args.checkpoint.is_file():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_args = make_args(Namespace(**saved["config"], output=args.checkpoint.parent), device)
    model = build_model(build_args, 74, 768, 35).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    ids, texts, audios, visions = [], [], [], []
    for path in files:
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
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            part = {key: value[start:start + args.batch_size].to(device)
                    for key, value in encoded.items()}
            vectors.append(bert(**part).last_hidden_state.cpu().numpy())
    del bert
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batch = prepare_unlabeled(np.concatenate(vectors), np.stack(audios),
                              np.stack(visions),
                              encoded["attention_mask"].numpy().astype(bool),
                              saved["scales"])
    rows = []
    with torch.inference_mode():
        for start in range(0, len(ids), args.batch_size):
            piece = {key: value[start:start + args.batch_size]
                     for key, value in batch.items()}
            features, mask, union, availability = ebmc_inputs(piece, device)
            out = model(features, mask, union, first_stage=False, label=None,
                        do_cf=False, availability=availability)
            probabilities = out[1][:, 0].softmax(dim=-1).cpu().numpy()
            intensity = (model.nlp_reg_head(out[0][:, 0]).squeeze(-1)
                         .tanh() * 3).cpu().numpy()
            observed = availability.cpu().numpy().astype(int)
            for index in range(len(probabilities)):
                class_id = int(probabilities[index].argmax())
                rows.append([ids[start + index], class_id, LABELS[class_id],
                             round(float(intensity[index]), 6),
                             *[round(float(value), 6) for value in probabilities[index]],
                             *observed[index].tolist()])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "class_id", "annotation", "regression_label",
                         "prob_negative", "prob_neutral", "prob_positive",
                         "audio_available", "text_available", "vision_available"])
        writer.writerows(rows)
    print(f"saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
