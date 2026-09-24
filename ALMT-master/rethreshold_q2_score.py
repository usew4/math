"""Click Run Python File after training: classify saved scores with ±0.2."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from rethreshold_q2_score import main


if __name__ == "__main__":
    main()
