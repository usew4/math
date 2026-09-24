"""One-click evaluation of the selected audio-convolution Q2 checkpoint."""
from __future__ import annotations

import sys
from pathlib import Path

from evaluate_q2_reliability import main


if __name__ == "__main__":
    if "--checkpoint" not in sys.argv:
        checkpoint = (Path(__file__).resolve().parent / "runs" /
                      "q2_text_anchor_audio_conv" / "best.pt")
        sys.argv[1:1] = ["--checkpoint", str(checkpoint)]
    main()
