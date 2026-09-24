"""Click Run Python File to train the two-head ALMT experiment."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from train_q2_multitask import main


if __name__ == "__main__":
    main()
