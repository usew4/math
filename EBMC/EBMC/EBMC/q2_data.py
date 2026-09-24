import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

COMBINATIONS = ('T', 'A', 'V', 'TA', 'TV', 'AV', 'TAV')
MODALITIES = (('audio', 'A', 74), ('text', 'T', 768), ('vision', 'V', 35))


def load_splits(path, splits=('train', 'valid'), limit=0):
    with open(path, 'rb') as f:
        source = pickle.load(f)
    result = {}
    for split in splits:
        p = source[split]
        n = len(p['id'])
        y = np.asarray(p['regression_labels']).reshape(-1)
        c = np.asarray(p['classification_labels']).reshape(-1)
        expected = (np.sign(y) + 1).astype(np.int64)
        if not np.isfinite(y).all() or np.any(np.abs(y) > 3) or not np.array_equal(c, expected):
            raise ValueError(f'{split}: invalid/inconsistent labels')
        lengths = {'audio': np.asarray(p['audio_lengths']),
                   'vision': np.asarray(p['vision_lengths']),
                   'text': np.asarray(p['text_bert'])[:, 1].sum(-1)}
        for key, _, dim in MODALITIES:
            x = p[key]
            if x.shape != (n, 50 if key == 'text' else 500, dim):
                raise ValueError(f'{split}/{key}: unexpected shape {x.shape}')
            if not np.isfinite(x).all():
                raise ValueError(f'{split}/{key}: nonfinite features')
            ls = lengths[key]
            if ls.shape != (n,) or np.any(ls != ls.astype(int)) or np.any(ls < 0) or np.any(ls > x.shape[1]):
                raise ValueError(f'{split}/{key}: invalid lengths')
        count = min(n, limit) if limit else n
        result[split] = {key: np.asarray(p[key][:count], dtype=np.float32).copy()
                         for key, _, _ in MODALITIES}
        result[split].update(lengths={k: v[:count].astype(int).copy() for k, v in lengths.items()},
                             y=y[:count].astype(np.float32).copy(),
                             cls=expected[:count].copy(), ids=list(p['id'][:count]))
    return result


def erase_interval(length, text, rate, rng, position='random'):
    start, end = (1, max(1, length - 1)) if text else (0, length)
    available = end - start
    if available < 2:
        return None
    size = min(available - 1, max(1, int(np.floor(rate * available + 0.5))))
    room = available - size
    offset = {'front': 0, 'middle': room // 2, 'back': room}.get(position)
    if offset is None:
        offset = int(rng.integers(room + 1))
    return start + offset, start + offset + size


class Q2Dataset(Dataset):
    def __init__(self, data, config, train=False):
        self.data, self.config, self.train = data, config, train
        self.epoch = 0
        self.scenario = ('complete', 0.0, 'random')

    def __len__(self):
        return len(self.data['y'])

    def __getitem__(self, index):
        combo, rate, position = self.scenario
        # No global RNG or worker-order dependence; validation is epoch-independent.
        tag = 0 if combo == 'complete' else COMBINATIONS.index(combo) + 1
        rng = np.random.default_rng(np.random.SeedSequence([
            self.config['seed'], self.epoch if self.train else 0, index,
            0 if self.train else tag, 0 if self.train else int(rate * 10000)]))
        if self.train:
            combo = 'complete' if rng.random() < self.config['complete_probability'] else rng.choice(COMBINATIONS)
        arrays, valid, observed, skipped = [], [], [], 0
        for key, letter, dim in MODALITIES:
            length = int(self.data['lengths'][key][index])
            x = np.zeros((500, dim), np.float32)
            x[:length] = self.data[key][index, :length]
            v = (np.arange(500) < length).astype(np.float32)
            o = v.copy()
            if combo != 'complete' and letter in combo:
                r = rng.uniform(*self.config['train_rates']) if self.train else rate
                span = erase_interval(length, key == 'text', r, rng, position)
                if span is None:
                    skipped += 1
                else:
                    a, b = span
                    x[a:b] = 0
                    o[a:b] = 0
            arrays.append(x); valid.append(v); observed.append(o)
        return dict(features=np.concatenate(arrays, -1), valid=np.stack(valid, -1),
                    observed=np.stack(observed, -1), y=self.data['y'][index],
                    cls=self.data['cls'][index], id=self.data['ids'][index], skipped=skipped)


def collate_q2(items):
    batch = {key: torch.from_numpy(np.stack([x[key] for x in items]))
             for key in ('features', 'valid', 'observed', 'y', 'cls')}
    batch['umask'] = batch['valid'].any(-1).float()
    for key in ('features', 'valid', 'observed'):
        batch[key] = batch[key].transpose(0, 1).contiguous()
    batch['ids'] = [x['id'] for x in items]
    batch['skipped'] = sum(x['skipped'] for x in items)
    return batch
