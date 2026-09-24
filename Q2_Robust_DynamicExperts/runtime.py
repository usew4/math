"""Local dependency bootstrap; no imports from the original project."""
import os
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
if importlib.util.find_spec('transformers') is None:
    sys.path.insert(0, str(ROOT / '.deps'))

import random
import numpy as np
import torch
import yaml


def settings(path=None):
    config = yaml.safe_load(Path(path or ROOT / 'config.yaml').read_text(encoding='utf-8'))
    assert 0 < config['training']['eval_batch_size']
    assert all(0 < r < 1 for r in config['missing']['rates'])
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment(config):
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    amp = config['training']['amp'] and device.type == 'cuda'
    return device, dtype, amp


def output_directory():
    path = ROOT / 'results'
    path.mkdir(exist_ok=True)
    return path
