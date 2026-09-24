"""Click Run in VS Code: ordinary three-class head without simulated missingness."""
import sys
from pathlib import Path

from train_q2 import main

project = Path(__file__).resolve().parent.parent
sys.argv = [str(Path(__file__)), "--feature-mode", "aligned", "--head-type", "flat",
            "--clean-only", "--validation-only", "--output",
            str(project / "runs" / "decalign_aligned_flat_clean")]
main()
