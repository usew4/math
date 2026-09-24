"""Encode Attachment 2 text_bert token fields with the frozen local BERT."""
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
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "text_bert" / "features.npz")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Small interface check only; omit for training")
    args = parser.parse_args()
    check_source_paths(need_bert=True)
    os.environ.setdefault("USE_TF", "0")
    # Import torch from the selected PyTorch environment before local Transformers.
    sys.path.append(str(BASE_SITE_PACKAGES))
    sys.path.insert(0, str(BERT_VENDOR_DIR))
    from transformers import AutoModel

    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert = AutoModel.from_pretrained(str(args.model_dir)).to(device).eval()
    with args.source.open("rb") as stream:
        source = pickle.load(stream)
    assert set(source) == {"train", "valid", "test"}
    arrays = {}
    report = {"source":str(args.source), "input_field":"text_bert",
              "model_dir":str(args.model_dir), "device":str(device),
              "bert_frozen":True, "batch_size":args.batch_size,
              "limit":args.limit, "splits":{}}
    for split in ("train", "valid", "test"):
        encoded = np.asarray(source[split]["text_bert"])
        if args.limit is not None:
            encoded = encoded[:args.limit]
        if encoded.ndim != 3 or encoded.shape[1:] != (3, 50):
            raise ValueError(f"Unexpected text_bert shape in {split}: {encoded.shape}")
        vectors = []
        with torch.inference_mode():
            for start in range(0, len(encoded), args.batch_size):
                part = encoded[start:start+args.batch_size]
                batch = {
                    "input_ids":torch.from_numpy(part[:, 0, :].astype(np.int64)).to(device),
                    "attention_mask":torch.from_numpy(part[:, 1, :].astype(np.int64)).to(device),
                    "token_type_ids":torch.from_numpy(part[:, 2, :].astype(np.int64)).to(device),
                }
                vectors.append(bert(**batch).last_hidden_state.cpu().numpy().astype(np.float16))
                if start % (args.batch_size*20) == 0:
                    print(f"{split}: {start+len(part)}/{len(encoded)}", flush=True)
        arrays[f"{split}_text"] = np.concatenate(vectors)
        arrays[f"{split}_mask"] = encoded[:, 1, :].astype(bool)
        report["splits"][split] = {"samples":len(encoded),
                                    "shape":list(arrays[f"{split}_text"].shape),
                                    "valid_token_positions":int(arrays[f"{split}_mask"].sum())}
        print(f"{split}: {len(encoded)} encoded", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
