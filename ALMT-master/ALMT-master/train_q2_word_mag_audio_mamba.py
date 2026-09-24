"""One-click Q2 training with word MAG and an audio temporal Mamba block."""
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
    from train_q2_reliability import main

    if "--config" not in sys.argv:
        config = (Path(__file__).resolve().parent / "configs" /
                  "q2_reliability_word_mag_audio_mamba.yaml")
        sys.argv[1:1] = ["--config", str(config)]
    main()
