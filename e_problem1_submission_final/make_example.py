"""Build a checkable text/audio/video example from one source clip."""
from __future__ import annotations

import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
import imageio_ffmpeg

OUTPUT = ROOT / "output_final"
SAMPLE = "-3g5yACwYnA__13"
with (OUTPUT / "run_metadata.json").open(encoding="utf-8") as f:
    source = Path(json.load(f)["source_zip"])
with (OUTPUT / "word_alignment.csv").open(encoding="utf-8-sig", newline="") as f:
    words = [r for r in csv.DictReader(f) if r["sample_id"] == SAMPLE]
with np.load(OUTPUT / "features" / f"{SAMPLE}.npz") as d:
    times = d["bin_center_sec"].copy()
    audio = d["audio"][:, 13].copy()
    speech = d["speech_mask"].copy()
    face = d["visual"][:, 12].copy()
    names = json.loads((OUTPUT / "visual_feature_names.json").read_text(encoding="utf-8"))
    smile = (d["visual"][:, names.index("mouthSmileLeft")] +
             d["visual"][:, names.index("mouthSmileRight")]) / 2

targets = [0.75, 2.25, 3.75, 4.75]
thumbnails = {}
with ZipFile(source) as z:
    name = next(x for x in z.namelist() if x.endswith("/-3g5yACwYnA/13.mp4"))
    with tempfile.TemporaryDirectory(prefix="e_example_") as td:
        path = Path(td) / "example.mp4"
        path.write_bytes(z.read(name))
        stream = imageio_ffmpeg.read_frames(str(path), output_params=["-vf", "fps=2,scale=320:-2"])
        meta = next(stream)
        size = meta["size"]
        for j, raw in enumerate(stream):
            time = (j + 0.5) / 2
            for target in targets:
                if abs(time - target) < 0.001:
                    thumbnails[target] = np.asarray(Image.frombytes("RGB", size, raw))
        stream.close()

fig = plt.figure(figsize=(13, 8), layout="constrained")
gs = fig.add_gridspec(4, 4, height_ratios=[1.25, 1, 1, 1.4])
ax_text = fig.add_subplot(gs[0, :])
ax_audio = fig.add_subplot(gs[1, :], sharex=ax_text)
ax_visual = fig.add_subplot(gs[2, :], sharex=ax_text)
for i, r in enumerate(words):
    start = float(r["start_sec"])
    end = float(r["end_sec"])
    color = "#245a92" if r["method"] == "pocketsphinx_forced" else "#bb6b25"
    ax_text.plot([start, end], [i % 3] * 2, color=color, linewidth=4, solid_capstyle="round")
    ax_text.text((start + end) / 2, i % 3 + 0.15, r["token"], rotation=35, ha="center", va="bottom", fontsize=8)
ax_text.set_ylim(-0.3, 3.25)
ax_text.set_yticks([])
ax_text.set_title("Word alignment: blue = acoustic forced; orange = interpolated / fallback")
ax_audio.plot(times, audio, color="#b15d24", linewidth=1.5)
ax_audio.fill_between(times, 0, speech * max(float(audio.max()), 0.1), color="#b15d24", alpha=0.15)
ax_audio.set_ylabel("Audio log RMS")
ax_visual.step(times, face, where="mid", color="#26704d", label="face rate")
ax_visual.plot(times, smile, color="#a63875", label="mouth smile")
ax_visual.legend(loc="upper right", fontsize=8)
ax_visual.set_ylabel("Visual coefficients")
ax_visual.set_xlabel("Common video timeline (seconds)")
for j, target in enumerate(targets):
    ax = fig.add_subplot(gs[3, j])
    ax.imshow(thumbnails[target])
    ax.set_title(f"Frame {target:.2f} s")
    ax.axis("off")
fig.suptitle(f"E problem 1 example: {SAMPLE}", fontsize=15)
fig.savefig(OUTPUT / "example_text_audio_video.png", dpi=170)
plt.close(fig)
print(OUTPUT / "example_text_audio_video.png")
