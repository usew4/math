"""Evaluation metrics and fixed local-gap experiments."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import corrupt


def metrics(labels, predictions, scores, predicted_scores):
    y, p = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    s, z = np.asarray(scores, dtype=float), np.asarray(predicted_scores, dtype=float)
    c = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    precision = np.diag(c) / np.maximum(c.sum(0), 1)
    recall = np.diag(c) / np.maximum(c.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        'n': len(y), 'accuracy': float((y == p).mean()),
        'macro_f1': float(f1.mean()),
        'recall': dict(zip(('Negative', 'Neutral', 'Positive'), recall.tolist())),
        'mae': float(np.abs(s - z).mean()),
        'pearson': float(np.corrcoef(s, z)[0, 1]) if np.std(s) > 0 and np.std(z) > 0 else None,
        'confusion': c.tolist(),
    }


@torch.inference_mode()
def evaluate(model, dataset, config, device, dtype, amp, condition=None, max_batches=0):
    model.eval()
    loader = DataLoader(dataset, batch_size=config['training']['eval_batch_size'], shuffle=False)
    ys, ps, ss, zs, rows = [], [], [], [], []
    removed, available = dict.fromkeys('TAV', 0), dict.fromkeys('TAV', 0)
    for step, (v, a, t, y, s, indices) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        ids = [dataset.ids[int(i)] for i in indices]
        if condition:
            v, a, t, records = corrupt(v, a, t, ids=ids, **condition)
            for record in records:
                for modality in 'TAV':
                    removed[modality] += record['removed'][modality]
                    available[modality] += record['available'][modality]
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            logits, scores = model(v.to(device), a.to(device), t.to(device))[:2]
        probs = logits.float().softmax(-1).cpu()
        pred = probs.argmax(-1)
        score = scores.float().cpu()
        ys.extend(y.tolist())
        ps.extend(pred.tolist())
        ss.extend(s.tolist())
        zs.extend(score.tolist())
        for sample_id, yy, pp, sz, zz, pr in zip(
            ids, y.tolist(), pred.tolist(), s.tolist(), score.tolist(), probs.tolist()
        ):
            rows.append({
                'id': sample_id, 'true_class': yy, 'pred_class': pp,
                'true_score': sz, 'pred_score': zz,
                'p_negative': pr[0], 'p_neutral': pr[1], 'p_positive': pr[2],
            })
    result = metrics(ys, ps, ss, zs)
    if condition:
        result['realized_additional_missing_rate'] = {
            m: removed[m] / max(1, available[m]) for m in 'TAV'
        }
    return result, rows
