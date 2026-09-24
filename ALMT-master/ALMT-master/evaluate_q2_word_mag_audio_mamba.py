"""Evaluate the selected word MAG plus audio Mamba checkpoint."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    if importlib.util.find_spec("mamba_ssm") is None:
        mamba_python = Path("D:/Anaconda_envs/envs/mamba/python.exe")
        if not mamba_python.is_file():
            raise FileNotFoundError(f"Mamba interpreter not found: {mamba_python}")
        raise SystemExit(subprocess.call(
            [str(mamba_python), str(Path(__file__).resolve()), *sys.argv[1:]]))
    from evaluate_q2_reliability import main

    if "--checkpoint" not in sys.argv:
        checkpoint = (Path(__file__).resolve().parent / "runs" /
                      "q2_text_anchor_word_mag_audio_mamba" / "best.pt")
        sys.argv[1:1] = ["--checkpoint", str(checkpoint)]
    main()
