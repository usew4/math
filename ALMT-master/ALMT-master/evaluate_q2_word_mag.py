"""One-click held-out evaluation of the selected word-aligned MAG experiment."""
from __future__ import annotations

import sys
from pathlib import Path

from evaluate_q2_reliability import main


if __name__ == "__main__":
    if "--checkpoint" not in sys.argv:
        checkpoint = (Path(__file__).resolve().parent / "runs" /
                      "q2_text_anchor_word_mag" / "best.pt")
        sys.argv[1:1] = ["--checkpoint", str(checkpoint)]
    main()
