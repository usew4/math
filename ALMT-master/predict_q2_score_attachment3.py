"""Click Run Python File to score the 30 aligned Attachment-3 samples."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from predict_q2_score_attachment3 import main


if __name__ == "__main__":
    main()
