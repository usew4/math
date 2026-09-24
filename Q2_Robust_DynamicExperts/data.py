"""Trusted competition pickle loader and reproducible contiguous masks."""
import pickle
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset

PATTERNS = ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV')


class Samples(Dataset):
    def __init__(self, entry, prefix='sample', labelled=True):
        self.text = torch.from_numpy(np.array(entry['text_bert'], dtype=np.int64, copy=True))
        self.audio = torch.from_numpy(np.nan_to_num(np.array(entry['audio'], dtype=np.float32, copy=True), nan=0, posinf=0, neginf=0))
        self.vision = torch.from_numpy(np.nan_to_num(np.array(entry['vision'], dtype=np.float32, copy=True), nan=0, posinf=0, neginf=0))
        n = len(self.text)
        self.ids = [str(x) for x in entry.get('id', [f'{prefix}:{i}' for i in range(n)])]
        self.labelled = labelled
        if self.text.shape != (n, 3, 50) or self.audio.shape != (n, 50, 74) or self.vision.shape != (n, 50, 35):
            raise ValueError('Expected aligned_50 input shapes')
        if len(self.ids) != n:
            raise ValueError('ID/sample count mismatch')
        self.labels = torch.from_numpy(np.array(entry['classification_labels'], dtype=np.int64).reshape(-1)) if labelled else torch.full((n,), -1)
        self.scores = torch.from_numpy(np.array(entry['regression_labels'], dtype=np.float32).reshape(-1)) if labelled else torch.zeros(n)
        if len(self.labels) != n or len(self.scores) != n:
            raise ValueError('Label/sample count mismatch')
        if labelled and (not torch.isfinite(self.scores).all() or not ((self.labels >= 0) & (self.labels <= 2)).all()):
            raise ValueError('Invalid labels')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, i):
        return self.vision[i], self.audio[i], self.text[i], self.labels[i], self.scores[i], i


def load_data(path):
    with open(path, 'rb') as f:
        source = pickle.load(f)
    return {split: Samples(source[split], split) for split in ('train', 'valid', 'test')}


def masks_and_domain(vision, audio, text):
    """Keep original slot indices; internal zeros never shorten the time axis.

    SEP determines the content interval. Without SEP, union support provides only
    an estimated endpoint; missing all-modal boundary slots cannot be recovered.
    """
    ids = text[:, 0]
    tmask = text[:, 1].bool() & (ids != 0)
    content = tmask & (ids != 101) & (ids != 102)
    amask = audio.abs().sum(-1) > 0
    vmask = vision.abs().sum(-1) > 0
    pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
    sep = ((ids == 102) * pos).amax(1)
    end = torch.where(sep > 0, sep - 1, ((content | amask | vmask) * pos).amax(1))
    domain = (pos > 0) & (pos <= end[:, None])
    return tmask, content, amask, vmask, domain


def corrupt(vision, audio, text, rng=None, pattern=None, rate=None,
            rates=(.1, .3, .5), ids=None, fixed_seed=None,
            position='random', synchronous=False):
    """One contiguous span per chosen modality. Padding/special tokens excluded."""
    v, a, t = vision.clone(), audio.clone(), text.clone()
    _, tm, am, vm, domain = masks_and_domain(vision, audio, text)
    observed = {'T': tm & domain, 'A': am & domain, 'V': vm & domain}
    rng = rng or np.random.default_rng()
    records = []
    for i in range(len(t)):
        gen = rng
        if fixed_seed is not None:
            payload = f'{fixed_seed}|{ids[i]}|{pattern}|{rate}|{position}|{synchronous}'.encode()
            gen = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little'))
        selected = pattern or str(gen.choice(PATTERNS))
        r = float(rate if rate is not None else gen.choice(rates))
        slots = torch.where(domain[i])[0]
        length = len(slots)
        width = min(max(1, round(r * length)), max(0, length - 1))
        max_start = length - width
        shared_start = int(gen.integers(max_start + 1)) if length else 0
        row = {'pattern': selected, 'target_rate': r, 'spans': {}, 'removed': {}, 'available': {}}
        for m in 'TAV':
            available = int(observed[m][i].sum())
            row['available'][m] = available
            row['removed'][m] = 0
            if m not in selected or width == 0 or available <= 1:
                continue
            if position == 'front': start = 0
            elif position == 'middle': start = max_start // 2
            elif position == 'back': start = max_start
            else: start = shared_start if synchronous else int(gen.integers(max_start + 1))
            span = slots[start:start + width]
            # Preserve at least one originally observed position in this modality.
            while len(span) and int(observed[m][i, span].sum()) >= available:
                span = span[:-1]
            if not len(span): continue
            row['removed'][m] = int(observed[m][i, span].sum())
            row['spans'][m] = [int(span[0]), int(span[-1]) + 1]
            if m == 'T': t[i, :, span] = 0
            elif m == 'A': a[i, span] = 0
            else: v[i, span] = 0
        records.append(row)
    return v, a, t, records
