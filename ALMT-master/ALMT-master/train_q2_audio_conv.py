"""One-click Q2 training with audio temporal convolution and masked attention."""
from __future__ import annotations

import sys
from pathlib import Path

from train_q2_reliability import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        config = Path(__file__).resolve().parent / "configs" / "q2_reliability_audio_conv.yaml"
        sys.argv[1:1] = ["--config", str(config)]
    main()
