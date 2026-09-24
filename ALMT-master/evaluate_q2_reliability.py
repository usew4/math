"""Click Run Python File after choosing the text-anchored model."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from evaluate_q2_reliability import main


if __name__ == "__main__":
    main()
