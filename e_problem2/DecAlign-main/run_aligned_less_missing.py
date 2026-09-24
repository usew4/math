"""Click Run in VS Code: aligned DecAlign with milder missing-data augmentation."""
import sys
from pathlib import Path

from train_q2 import main

project = Path(__file__).resolve().parent.parent
sys.argv = [str(Path(__file__)), "--feature-mode", "aligned", "--robust-prob", "0.3",
            "--validation-only", "--output", str(project / "runs" / "decalign_aligned_less_missing")]
main()
