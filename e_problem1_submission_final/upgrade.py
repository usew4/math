"""Question 1 upgrade: audio forced alignment and optional face blendshapes.

Reads the original 100-video ZIP and the reproducible baseline `output/`.
Unknown dictionary words receive explicit interpolation flags; no hidden imputation.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import io
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from extract import (AUDIO_RATE, DEFAULT_ZIP, N_STEPS, ROOT, WORDS, decode_audio,
                     hash_vector, imageio_ffmpeg, load_manifest, safe_name,
                     SentimentIntensityAnalyzer)

sys.path.insert(0, str(ROOT / "vendor"))
from pocketsphinx import Decoder, get_model_path


def make_decoder(temp: Path) -> Decoder:
    # PocketSphinx's Windows C library cannot open a non-ASCII model path.
    p = temp / "model"
    shutil.copytree(get_model_path(), p)
    return Decoder(samprate=AUDIO_RATE, hmm=str(p / "en-us" / "en-us"),
                   dict=str(p / "en-us" / "cmudict-en-us.dict"),
                   lm=str(p / "en-us" / "en-us.lm.bin"))


def normalize_word(word: str) -> str:
    return word.lower().replace("’", "'").replace("`", "'").replace("–", "-")


def align_words(decoder: Decoder, wave: np.ndarray, tokens: list[str], duration: float) -> tuple[list[dict], dict]:
    norm = [normalize_word(t) for t in tokens]
    known_ids = [i for i, t in enumerate(norm) if decoder.lookup_word(t)]
    unknown_ids = [i for i in range(len(norm)) if i not in set(known_ids)]
    stats = {"dictionary_words": len(known_ids), "oov_words": len(unknown_ids),
             "forced_words": 0, "alignment_ok": False, "error": ""}
    anchors: dict[int, tuple[float, float]] = {}
    if known_ids:
        try:
            decoder.set_align_text(" ".join(norm[i] for i in known_ids))
            decoder.start_utt()
            pcm = np.clip(wave * 32768, -32768, 32767).astype("<i2").tobytes()
            decoder.process_raw(pcm, False, True)
            decoder.end_utt()
            segments = [s for s in decoder.seg() if not s.word.startswith("<")]
            if len(segments) == len(known_ids):
                pairs = list(zip(known_ids, segments))
            else:
                actual = [re.sub(r"\(\d+\)$", "", s.word).lower() for s in segments]
                expected = [norm[i] for i in known_ids]
                matcher = difflib.SequenceMatcher(a=expected,b=actual,autojunk=False)
                pairs = [(known_ids[a+k],segments[b+k]) for a,b,n in matcher.get_matching_blocks()
                         for k in range(n)]
                if len(pairs) < max(1,len(known_ids)//2):
                    raise ValueError(f"only {len(pairs)}/{len(known_ids)} segments matched")
            for idx, seg in pairs:
                anchors[idx] = (max(0., seg.start_frame * .01),
                                min(duration, (seg.end_frame + 1) * .01))
            stats["forced_words"] = len(anchors)
            stats["alignment_ok"] = True
        except Exception as ex:
            stats["error"] = str(ex)[:160]
            anchors.clear()
    rows: list[dict] = []
    n = len(tokens)
    for i, token in enumerate(tokens):
        if i in anchors:
            start, end = anchors[i]
            method = "pocketsphinx_forced"
        else:
            before = max((j for j in anchors if j < i), default=None)
            after = min((j for j in anchors if j > i), default=None)
            left = anchors[before][1] if before is not None else 0.
            right = anchors[after][0] if after is not None else duration
            a = before + 1 if before is not None else 0
            b = after if after is not None else n
            weights = [max(1, len(norm[j].replace("'", ""))) for j in range(a, b)]
            total = max(1, sum(weights))
            start = left + (right - left) * sum(weights[:i-a]) / total
            end = left + (right - left) * sum(weights[:i-a+1]) / total
            method = "oov_or_partial_interpolation" if stats["alignment_ok"] else "uniform_fallback"
        start = float(np.clip(start, 0, duration))
        end = float(np.clip(end, start, duration))
        rows.append({"word_index": i, "token": token, "start_sec": round(start, 4),
                     "end_sec": round(end, 4), "method": method,
                     "bin": min(N_STEPS-1, int((start+end)/2 / max(duration,1e-6)*N_STEPS))})
    return rows, stats


def aligned_text_features(tokens: list[str], alignment: list[dict], duration: float,
                          analyzer: SentimentIntensityAnalyzer) -> np.ndarray:
    out = np.zeros((N_STEPS, 37), dtype=np.float32)
    weight = np.zeros(N_STEPS, dtype=np.float32)
    for i, (token, record) in enumerate(zip(tokens, alignment)):
        # A short context window lets VADER account for nearby negation/intensifiers.
        phrase = " ".join(tokens[max(0,i-2):min(len(tokens),i+3)])
        score = analyzer.polarity_scores(phrase)
        vec = np.zeros(36, dtype=np.float32)
        vec[:32] = hash_vector(token)
        vec[32:] = [score["neg"], score["neu"], score["pos"], score["compound"]]
        a, b = record["start_sec"], record["end_sec"]
        if b <= a:
            b = min(duration, a + .01)
        for k in range(N_STEPS):
            left, right = duration*k/N_STEPS, duration*(k+1)/N_STEPS
            overlap = max(0., min(right,b)-max(left,a))
            if overlap > 0:
                out[k,:36] += vec*overlap
                weight[k] += overlap
                out[k,36] += 1
    nonzero = weight > 0
    out[nonzero,:36] /= weight[nonzero,None]
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(source: Path, baseline: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    features = out / "features"
    features.mkdir(exist_ok=True)
    summary, all_words = [], []
    with (baseline/"all_100_summary.csv").open(encoding="utf-8-sig",newline="") as f:
        old_summary = {r["sample_id"]:r for r in csv.DictReader(f)}
    with (baseline/"word_alignment_estimates.csv").open(encoding="utf-8-sig",newline="") as f:
        fallback: dict[str,list[dict]] = {}
        for r in csv.DictReader(f):
            fallback.setdefault(r["sample_id"],[]).append(r)
    with tempfile.TemporaryDirectory(prefix="e_q1_ascii_") as td:
        tmp = Path(td)
        decoder = make_decoder(tmp)
        with ZipFile(source) as z:
            manifest = load_manifest(z)
            videos = {f"{Path(n).parent.name}/{Path(n).stem}": n for n in z.namelist()
                      if n.lower().endswith(".mp4") and "附件1-" in n}
            if len(videos) != 100:
                raise ValueError(f"expected 100 MP4, found {len(videos)}")
            for j, row in enumerate(manifest, 1):
                vid, clip = str(row["video_id"]), str(row["clip_id"])
                name = safe_name(vid, clip)
                with np.load(baseline / "features" / f"{name}.npz") as base:
                    audio, visual = base["audio"].copy(), base["visual"].copy()
                    times = base["bin_center_sec"].copy()
                    speech = base["speech_mask"].copy()
                    visual_counts = base["visual_frame_count"].copy()
                duration = float(times[-1] * N_STEPS / (N_STEPS-.5))
                path = tmp / "clip.mp4"
                with z.open(videos[f"{vid}/{clip}"]) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                wave = decode_audio(imageio_ffmpeg.get_ffmpeg_exe(), path)
                path.unlink()
                tokens = WORDS.findall(str(row["text"] or ""))
                alignment, stats = align_words(decoder, wave, tokens, duration)
                if not stats["alignment_ok"]:
                    prior = fallback[name]
                    if len(prior) == len(tokens):
                        for a,p in zip(alignment,prior):
                            a["start_sec"] = float(p["estimated_start_sec"])
                            a["end_sec"] = float(p["estimated_end_sec"])
                            a["bin"] = int(p["bin"])
                            a["method"] = "vad_orthographic_fallback"
                text = aligned_text_features(tokens, alignment, duration, SentimentIntensityAnalyzer())
                np.savez_compressed(features / f"{name}.npz", text=text, audio=audio,
                                    visual=visual, bin_center_sec=times, speech_mask=speech,
                                    visual_frame_count=visual_counts)
                all_words.extend({"sample_id": name, **r} for r in alignment)
                summary.append({"sample_id": name, "video_id": vid, "clip_id": clip,
                                "label": row["label"], "annotation": row["annotation"],
                                "duration_sec": round(duration,3), "word_count": len(tokens),
                                "audio_duration_sec":old_summary[name]["audio_duration_sec"],
                                "source_fps":old_summary[name]["source_fps"],
                                "source_frames":old_summary[name]["source_frames"],
                                "speech_bins":old_summary[name]["speech_bins"],
                                "visual_frames_read":old_summary[name]["visual_frames_read"],
                                **stats, "face_frames": int(old_summary[name]["face_frames"]),
                                "text_shape": "50x37", "audio_shape": "50x17", "visual_shape": "50x12"})
                print(f"[{j:3d}/100] {name}: {stats['forced_words']}/{len(tokens)} forced, {stats['oov_words']} OOV", flush=True)
    # Preserve the frame-level provenance from the baseline visual extractor.
    shutil.copy2(baseline / "visual_frame_log.csv", out / "visual_frame_log.csv")
    write_csv(out / "all_100_summary.csv", summary)
    write_csv(out / "word_alignment.csv", all_words)
    metadata = {"source_zip":str(source), "baseline":str(baseline), "samples":100,
                "grid_steps":N_STEPS, "word_timing":"PocketSphinx 5.0.4 forced alignment with explicit OOV interpolation",
                "visual":"baseline scikit-image 0.23.2 LBP face detection, 2 fps",
                "text":"32 signed character trigram hash + 4 context VADER scores + count",
                "audio":"13 MFCC + log RMS + ZCR + spectral centroid + pitch proxy"}
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    print("COMPLETE",len(summary),"samples",flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,default=DEFAULT_ZIP)
    ap.add_argument("--baseline",type=Path,default=ROOT/"output")
    ap.add_argument("--output",type=Path,default=ROOT/"output_v2")
    args = ap.parse_args()
    run(args.source,args.baseline,args.output)
