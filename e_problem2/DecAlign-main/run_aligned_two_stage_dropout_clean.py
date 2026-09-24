"""Click Run: fair two-stage comparison, aligned data, no artificial missingness."""
import sys
from pathlib import Path

from train_q2 import main

project = Path(__file__).resolve().parent.parent
sys.argv = [str(Path(__file__)), "--feature-mode", "aligned", "--head-type", "hierarchical",
            "--clean-only", "--validation-only", "--output",
            str(project / "runs" / "decalign_aligned_two_stage_dropout_clean")]
main()
