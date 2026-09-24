"""Click Run Python File once the score model is selected to evaluate held-out test."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from evaluate_q2_score import main


if __name__ == "__main__":
    main()
