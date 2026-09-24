"""Click Run Python File: train ALMT score regression with neutral oversampling."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))
sys.argv[1:1] = ["--config", str(source_root / "configs" / "q2_score_neutral_sampling.yaml")]

from train_q2_score import main


if __name__ == "__main__":
    main()
