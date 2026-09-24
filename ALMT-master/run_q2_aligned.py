"""In VS Code, click Run Python File to train ALMT on E题问题二已对齐数据."""
import sys
from pathlib import Path

source_root = Path(__file__).resolve().parent / "ALMT-master"
sys.path.insert(0, str(source_root))

from train_q2_aligned import main


if __name__ == "__main__":
    main()
