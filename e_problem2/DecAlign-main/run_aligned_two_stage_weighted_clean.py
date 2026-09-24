"""Click Run: two-stage aligned DecAlign, no artificial missingness, mild class weights."""
import sys
from pathlib import Path

from train_q2 import main

project = Path(__file__).resolve().parent.parent
sys.argv = [str(Path(__file__)), "--feature-mode", "aligned", "--head-type", "hierarchical",
            "--class-weight", "sqrt_balanced", "--clean-only", "--validation-only",
            "--output", str(project / "runs" / "decalign_aligned_two_stage_weighted_clean")]
main()
