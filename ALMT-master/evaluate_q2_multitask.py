"""Click Run Python File after selecting the best two-head ALMT checkpoint."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from evaluate_q2_multitask import main


if __name__ == "__main__":
    main()
