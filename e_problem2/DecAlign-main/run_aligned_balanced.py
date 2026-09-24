"""Click Run in VS Code: milder missing augmentation plus mild class balancing."""
import sys
from pathlib import Path

from train_q2 import main

project = Path(__file__).resolve().parent.parent
sys.argv = [str(Path(__file__)), "--feature-mode", "aligned", "--robust-prob", "0.3",
            "--class-weight", "sqrt_balanced", "--validation-only", "--output",
            str(project / "runs" / "decalign_aligned_balanced")]
main()
