"""Problem E, question 1: reproducible features for all 100 supplied clips.

The source ZIP is read without extracting the multi-gigabyte other attachments.
Word timing is an explicitly approximate speech-activity allocation, not forced alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from openpyxl import load_workbook
from scipy.fft import dct
from scipy.signal import stft
from scipy.ndimage import laplace
from PIL import Image
from skimage.data import lbp_frontal_face_cascade_filename
from skimage.feature import Cascade

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import imageio_ffmpeg  # noqa: E402
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: E402

DEFAULT_ZIP = Path(r"D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据.zip")
N_STEPS = 50
AUDIO_RATE = 16000
VISUAL_FPS = 2
WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?")


def safe_name(video_id: str, clip_id: str) -> str:
    return f"{video_id}__{clip_id}"


def hash_vector(token: str, dim: int = 32) -> np.ndarray:
    """Fixed signed character-trigram hashing, independent of corpus or labels."""
    out = np.zeros(dim, dtype=np.float32)
    s = f"<{token.lower()}>"
    for i in range(max(1, len(s) - 2)):
        tri = s[i : i + 3]
        digest = hashlib.blake2b(tri.encode(), digest_size=4).digest()
        h = int.from_bytes(digest, "little")
        out[h % dim] += 1.0 if h & 32 else -1.0
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def mel_filterbank(sr: int, nfft: int, count: int = 26) -> np.ndarray:
    def hz_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    hz = mel_hz(np.linspace(hz_mel(0), hz_mel(sr / 2), count + 2))
    bins = np.floor((nfft + 1) * hz / sr).astype(int)
    fb = np.zeros((count, nfft // 2 + 1), dtype=np.float32)
    for j in range(count):
        left, mid, right = bins[j : j + 3]
        for k in range(left, mid):
            if mid > left and k < fb.shape[1]:
                fb[j, k] = (k - left) / (mid - left)
        for k in range(mid, right):
            if right > mid and k < fb.shape[1]:
                fb[j, k] = (right - k) / (right - mid)
    return fb


def decode_audio(ffmpeg: str, path: Path) -> np.ndarray:
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-500:])
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768


def audio_features(wave: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """13 MFCC + log RMS + ZCR + centroid + pitch, 50 time bins."""
    result = np.zeros((N_STEPS, 17), dtype=np.float32)
    activity = np.zeros(N_STEPS, dtype=np.float32)
    if len(wave) < 512:
        return result, activity
    nfft, hop = 512, 160
    freqs, times, spec = stft(wave, fs=AUDIO_RATE, nperseg=nfft, noverlap=nfft - hop, boundary=None)
    power = (np.abs(spec) ** 2).astype(np.float32)
    fb = mel_filterbank(AUDIO_RATE, nfft)
    mfcc = dct(np.log(np.maximum(fb @ power, 1e-10)), type=2, norm="ortho", axis=0)[:13].T
    rms = np.sqrt(np.maximum(power.sum(axis=0), 1e-12))
    center = (power * freqs[:, None]).sum(axis=0) / np.maximum(power.sum(axis=0), 1e-12)
    frame_extra = np.zeros((len(times), 4), dtype=np.float32)
    frame_extra[:, 0] = np.log1p(rms)
    frame_extra[:, 2] = center / 8000.0
    for j, t in enumerate(times):
        p = int(t * AUDIO_RATE)
        frame = wave[max(0, p - nfft // 2) : p + nfft // 2]
        if len(frame) < nfft // 2:
            continue
        frame_extra[j, 1] = np.mean(np.diff(np.signbit(frame)))
        if j % 5 == 0 and rms[j] > 0.01:
            x = frame - frame.mean()
            ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
            lo, hi = int(AUDIO_RATE / 400), min(int(AUDIO_RATE / 70), len(ac))
            if hi > lo:
                lag = lo + np.argmax(ac[lo:hi])
                if ac[lag] > 0.3 * ac[0]:
                    frame_extra[j, 3] = AUDIO_RATE / lag / 400.0
    full = np.concatenate([mfcc, frame_extra], axis=1)
    idx = np.clip((times / max(duration, 1e-6) * N_STEPS).astype(int), 0, N_STEPS - 1)
    for b in range(N_STEPS):
        mask = idx == b
        if mask.any():
            result[b] = np.mean(full[mask], axis=0)
            activity[b] = float(np.mean(rms[mask]))
    # A relative VAD threshold handles changing recording gain.
    positive = activity[activity > 0]
    threshold = max(0.004, float(np.percentile(positive, 35)) * 0.55) if len(positive) else 0.004
    activity = (activity > threshold).astype(np.float32)
    return result, activity


def distribute_words(words: list[str], speech_mask: np.ndarray, duration: float) -> list[tuple[str, float, float, int]]:
    """Allocate words to voiced bins by orthographic length; timestamps are estimates."""
    if not words:
        return []
    active = np.where(speech_mask > 0)[0]
    if not len(active):
        active = np.arange(N_STEPS)
    weights = np.array([max(1, len(w.replace("'", ""))) for w in words], dtype=float)
    centers = (np.cumsum(weights) - weights / 2) / weights.sum()
    order = np.clip((centers * len(active)).astype(int), 0, len(active) - 1)
    out = []
    for word, b in zip(words, active[order]):
        start = float(b / N_STEPS * duration)
        end = float((b + 1) / N_STEPS * duration)
        out.append((word, start, end, int(b)))
    return out


def text_features(text: str, activity: np.ndarray, duration: float, analyzer) -> tuple[np.ndarray, list[tuple[str, float, float, int]]]:
    words = WORDS.findall(text)
    mapping = distribute_words(words, activity, duration)
    out = np.zeros((N_STEPS, 37), dtype=np.float32)
    counts = np.zeros(N_STEPS, dtype=np.float32)
    for word, _, _, b in mapping:
        out[b, :32] += hash_vector(word)
        score = analyzer.polarity_scores(word)
        out[b, 32:36] += [score["neg"], score["neu"], score["pos"], score["compound"]]
        counts[b] += 1
    for b in range(N_STEPS):
        if counts[b]:
            out[b, :36] /= counts[b]
            out[b, 36] = counts[b]
    return out, mapping


def video_metadata(path: Path) -> dict:
    stream = imageio_ffmpeg.read_frames(str(path))
    try:
        return next(stream)
    finally:
        stream.close()


def visual_features(path: Path, duration: float, face_model: Cascade) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Frame-level face, lighting, color and motion descriptors."""
    out = np.zeros((N_STEPS, 12), dtype=np.float32)
    counts = np.zeros(N_STEPS, dtype=np.int16)
    frames = []
    previous_gray = None
    stream = imageio_ffmpeg.read_frames(str(path), output_params=["-vf", f"fps={VISUAL_FPS},scale=320:-2"])
    meta = next(stream)
    w, h = meta["size"]
    for j, raw in enumerate(stream):
        t = (j + 0.5) / VISUAL_FPS
        frame = np.asarray(Image.frombytes("RGB", (w, h), raw))
        gray = (frame[:, :, 0] * 0.299 + frame[:, :, 1] * 0.587 + frame[:, :, 2] * 0.114).astype(np.uint8)
        small = np.asarray(Image.fromarray(frame).resize((160, 90)))
        small_gray = (small[:, :, 0] * 0.299 + small[:, :, 1] * 0.587 + small[:, :, 2] * 0.114).astype(np.uint8)
        faces = face_model.detect_multi_scale(gray, scale_factor=1.2, step_ratio=1.5,
                                              min_size=(30, 30), max_size=(200, 200), min_neighbor_number=4)
        f = np.zeros(12, dtype=np.float32)
        f[0] = float(len(faces) > 0)
        f[1] = min(len(faces), 3) / 3
        f[7] = float(small_gray.mean() / 255)
        f[8] = float(np.mean((small.max(axis=2) - small.min(axis=2)) / np.maximum(small.max(axis=2), 1)))
        f[9] = float(min(laplace(small_gray.astype(float)).var() / 1000, 1))
        if previous_gray is not None:
            f[10] = float(np.mean(np.abs(small_gray.astype(float) - previous_gray.astype(float))) / 255)
        previous_gray = small_gray
        if len(faces):
            face = max(faces, key=lambda a: a["width"] * a["height"])
            x, y, fw, fh = face["c"], face["r"], face["width"], face["height"]
            f[2:6] = [(x + fw / 2) / w, (y + fh / 2) / h, fw / w, fh / h]
            face_gray = gray[y : y + fh, x : x + fw]
            f[6] = float(np.mean(face_gray[fh // 2 :]) / 255) if fh > 2 else 0
            f[11] = float(face_gray.mean() / 255)
        b = min(int(t / max(duration, 1e-6) * N_STEPS), N_STEPS - 1)
        out[b] += f
        counts[b] += 1
        frames.append({"time_sec": round(float(t), 3), "bin": int(b), "face_detected": int(f[0])})
    stream.close()
    nonzero = counts > 0
    out[nonzero] /= counts[nonzero, None]
    return out, counts, frames


def load_manifest(z: ZipFile) -> list[dict]:
    label_path = next(x for x in z.namelist() if x.endswith("label-100.xlsx"))
    wb = load_workbook(io.BytesIO(z.read(label_path)), read_only=True, data_only=True)
    ws = wb["label"]
    rows = list(ws.values)
    header = [str(x) for x in rows[0]]
    result = [dict(zip(header, row)) for row in rows[1:] if row[0] is not None]
    if len(result) != 100:
        raise ValueError(f"Expected 100 labeled clips, got {len(result)}")
    return result


def run(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    feature_dir = output / "features"
    feature_dir.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    analyzer = SentimentIntensityAnalyzer()
    face_model = Cascade(lbp_frontal_face_cascade_filename())
    summary, word_rows, frame_rows, errors = [], [], [], []
    with ZipFile(source) as z:
        manifest = load_manifest(z)
        video_paths = {x.filename: x for x in z.infolist() if x.filename.lower().endswith(".mp4") and "附件1-" in x.filename}
        by_id = {f"{Path(k).parent.name}/{Path(k).stem}": k for k in video_paths}
        if len(by_id) != 100:
            raise ValueError(f"Expected 100 source videos, got {len(by_id)}")
        for index, row in enumerate(manifest, 1):
            video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
            key = f"{video_id}/{clip_id}"
            name = safe_name(video_id, clip_id)
            if key not in by_id:
                raise FileNotFoundError(key)
            entry = video_paths[by_id[key]]
            with tempfile.TemporaryDirectory(prefix="e_problem1_") as td:
                path = Path(td) / f"{name}.mp4"
                with z.open(entry) as src, path.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                meta = video_metadata(path)
                fps = float(meta.get("fps") or 0)
                duration = float(meta.get("duration") or 0)
                frame_count = round(duration * fps) if fps > 0 else 0
                wave = decode_audio(ffmpeg, path)
                audio_duration = len(wave) / AUDIO_RATE
                if duration <= 0:
                    duration = audio_duration
                duration = max(duration, audio_duration)
                audio, speech = audio_features(wave, duration)
                text, mapping = text_features(str(row["text"] or ""), speech, duration, analyzer)
                visual, visual_counts, frames = visual_features(path, duration, face_model)
                times = (np.arange(N_STEPS) + 0.5) * duration / N_STEPS
                np.savez_compressed(feature_dir / f"{name}.npz", text=text, audio=audio, visual=visual,
                                    bin_center_sec=times.astype(np.float32), speech_mask=speech,
                                    visual_frame_count=visual_counts)
                word_rows.extend({"sample_id": name, "word_index": j, "token": w, "estimated_start_sec": round(a, 4),
                                  "estimated_end_sec": round(b, 4), "bin": bi, "method": "VAD_weighted_orthographic_approximation"}
                                 for j, (w, a, b, bi) in enumerate(mapping))
                frame_rows.extend({"sample_id": name, **f} for f in frames)
                summary.append({"sample_id": name, "video_id": video_id, "clip_id": clip_id, "annotation": row["annotation"],
                                "label": row["label"], "video_duration_sec": round(duration, 3),
                                "audio_duration_sec": round(audio_duration, 3), "source_fps": round(fps, 3),
                                "source_frames": int(frame_count), "word_count": len(mapping),
                                "speech_bins": int(speech.sum()), "visual_frames_read": len(frames),
                                "face_frames": sum(x["face_detected"] for x in frames),
                                "text_shape": "50x37", "audio_shape": "50x17", "visual_shape": "50x12",
                                "alignment": "50 equal-duration bins", "status": "ok"})
            print(f"[{index:3d}/100] {name} duration={duration:.2f}s words={len(mapping)} frames={len(frames)}", flush=True)
    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            raise ValueError(f"No records for {path.name}")
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    write_csv(output / "all_100_summary.csv", summary)
    write_csv(output / "word_alignment_estimates.csv", word_rows)
    write_csv(output / "visual_frame_log.csv", frame_rows)
    import scipy
    import skimage
    import PIL
    import openpyxl
    env = {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
           "scikit_image": skimage.__version__, "Pillow": PIL.__version__, "openpyxl": openpyxl.__version__,
           "imageio_ffmpeg": "0.6.0", "vaderSentiment": "3.3.2",
           "visual_detector": "scikit-image LBP frontal face cascade",
           "ffmpeg": ffmpeg, "source_zip": str(source), "samples": len(summary),
           "grid_steps": N_STEPS, "visual_sample_fps": VISUAL_FPS,
           "word_timing": "approximate VAD-based allocation; not forced alignment"}
    (output / "run_metadata.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print("COMPLETE", len(summary), "samples", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    run(args.source, args.output)
