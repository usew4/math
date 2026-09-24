"""Encode Attachment 2 raw_text with a frozen local BERT for Q2 training."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from data_unaligned import DEFAULT_SOURCE
from paths import BASE_SITE_PACKAGES, BERT_MODEL_DIR, BERT_VENDOR_DIR, check_source_paths


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-dir", type=Path, default=BERT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "raw_text_bert" / "features.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int,
                        help="For a small interface check only; omit for training")
    args = parser.parse_args()
    check_source_paths(need_bert=True)
    os.environ.setdefault("USE_TF", "0")
    # The selected PyTorch environment contains CUDA torch. The local vendor adds
    # Transformers, while Anaconda base supplies regex/openpyxl metadata.
    sys.path.append(str(BASE_SITE_PACKAGES))
    sys.path.insert(0, str(BERT_VENDOR_DIR))
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    bert = AutoModel.from_pretrained(str(args.model_dir)).to(device).eval()
    with args.source.open("rb") as stream:
        source = pickle.load(stream)
    assert set(source) == {"train", "valid", "test"}
    arrays = {}
    report = {"source":str(args.source), "model_dir":str(args.model_dir),
              "device":str(device), "bert_frozen":True, "batch_size":args.batch_size,
              "limit":args.limit, "splits":{}}
    for split in ("train", "valid", "test"):
        entry = source[split]
        texts = list(entry["raw_text"])
        if args.limit is not None:
            texts = texts[:args.limit]
        vectors, masks = [], []
        matches = 0
        with torch.inference_mode():
            for start in range(0, len(texts), args.batch_size):
                subset = texts[start:start+args.batch_size]
                encoded = tokenizer(subset, padding="max_length", truncation=True,
                                    max_length=50, return_tensors="pt")
                given_ids = np.asarray(entry["text_bert"][start:start+len(subset), 0, :])
                matches += int(np.all(encoded["input_ids"].numpy() == given_ids, axis=1).sum())
                batch = {key:value.to(device) for key,value in encoded.items()}
                hidden = bert(**batch).last_hidden_state.cpu().numpy()
                vectors.append(hidden.astype(np.float16))
                masks.append(encoded["attention_mask"].numpy().astype(bool))
                if start % (args.batch_size*20) == 0:
                    print(f"{split}: {start+len(subset)}/{len(texts)}", flush=True)
        arrays[f"{split}_text"] = np.concatenate(vectors)
        arrays[f"{split}_mask"] = np.concatenate(masks)
        report["splits"][split] = {"samples":len(texts),
                                    "token_ids_exact_match":matches,
                                    "shape":list(arrays[f"{split}_text"].shape)}
        print(f"{split}: {len(texts)} encoded, token IDs matched {matches}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
