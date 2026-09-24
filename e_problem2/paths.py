"""Local paths for the desktop Q2 project; environment variables may override them."""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTORCH_PYTHON = Path(r"D:\Anaconda_envs\envs\pytorch312\python.exe")
DATA_ROOT = Path(os.environ.get(
    "MOSEI_DATA_ROOT",
    r"D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\E题数据\E题数据",
))
UNALIGNED_PKL = Path(os.environ.get(
    "MOSEI_UNALIGNED_PKL",
    str(DATA_ROOT / "附件2-数据集特征文件" / "unaligned_50.pkl"),
))
ALIGNED_PKL = Path(os.environ.get(
    "MOSEI_ALIGNED_PKL",
    str(DATA_ROOT / "附件2-数据集特征文件" / "aligned_50.pkl"),
))
MISSING_UNALIGNED_DIR = Path(os.environ.get(
    "MOSEI_MISSING_UNALIGNED_DIR",
    str(DATA_ROOT / "附件3-模态缺失特征样本" / "未对齐版本"),
))
MISSING_ALIGNED_DIR = Path(os.environ.get(
    "MOSEI_MISSING_ALIGNED_DIR",
    str(DATA_ROOT / "附件3-模态缺失特征样本" / "对齐版本"),
))
BERT_ROOT = Path(os.environ.get(
    "MOSEI_BERT_ROOT",
    r"C:\Users\wwh\Documents\ChatGPT\数学建模\e_problem1",
))
BERT_MODEL_DIR = BERT_ROOT / "bert_model"
BERT_VENDOR_DIR = BERT_ROOT / "vendor_bert"
BASE_SITE_PACKAGES = Path(r"D:\anaconda\Lib\site-packages")


def check_source_paths(*, need_bert: bool = False, need_missing: bool = False) -> None:
    if not UNALIGNED_PKL.is_file():
        raise FileNotFoundError(f"未对齐训练数据不存在：{UNALIGNED_PKL}")
    if need_missing and not MISSING_UNALIGNED_DIR.is_dir():
        raise FileNotFoundError(f"附件3未对齐目录不存在：{MISSING_UNALIGNED_DIR}")
    if need_bert:
        if not (BERT_MODEL_DIR / "model.safetensors").is_file():
            raise FileNotFoundError(f"BERT 权重不存在：{BERT_MODEL_DIR}")
        if not (BERT_VENDOR_DIR / "transformers").is_dir():
            raise FileNotFoundError(f"Transformers 本地依赖不存在：{BERT_VENDOR_DIR}")


if __name__ == "__main__":
    import sys
    import torch

    check_source_paths(need_bert=True, need_missing=True)
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.cuda.is_available()}")
    print(f"训练数据: {UNALIGNED_PKL}")
    print(f"附件3: {MISSING_UNALIGNED_DIR}")
    print(f"BERT: {BERT_MODEL_DIR}")
