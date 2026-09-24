"""Click Run to predict every aligned attachment-3 sample with the retained model."""
import csv
import json
import pickle
import argparse
from pathlib import Path
import runtime
import torch
from torch.utils.data import DataLoader
from data import Samples
from model import load_model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    config = runtime.settings(args.config)
    output = runtime.output_directory()
    checkpoint = runtime.ROOT / 'checkpoints' / 'dynamic_experts_base.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    device, dtype, amp = runtime.environment(config)
    model, _ = load_model(device)
    model.eval()
    files = sorted(Path(config['attachment3']).glob('*.pkl'))
    if not files: raise FileNotFoundError(config['attachment3'])
    results = []
    for path in files:
        with path.open('rb') as f: obj = pickle.load(f)
        dataset = Samples(obj.get('test', obj), path.stem, labelled=False)
        for v, a, t, _, _, ix in DataLoader(dataset, batch_size=config['training']['eval_batch_size']):
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                logits, scores = model(v.to(device), a.to(device), t.to(device))[:2]
            probs = logits.float().softmax(-1).cpu()
            for i, score, prob in zip(ix.tolist(), scores.float().cpu().tolist(), probs.tolist()):
                cls = max(range(3), key=lambda c: prob[c])
                results.append({'source_file': path.name, 'sample_index': i, 'id': dataset.ids[i],
                                'prediction': ('Negative', 'Neutral', 'Positive')[cls], 'pred_score': score,
                                'prob_negative': prob[0], 'prob_neutral': prob[1], 'prob_positive': prob[2]})
    if len({(r['source_file'], r['sample_index']) for r in results}) != len(results):
        raise ValueError('Duplicate prediction rows')
    with (output / 'attachment3_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    summary = {'checkpoint': str(checkpoint), 'epoch': saved.get('epoch'), 'files': len(files), 'samples': len(results),
               'classes': {c: sum(r['prediction'] == c for r in results) for c in ('Negative', 'Neutral', 'Positive')},
               'note': 'Unlabelled final predictions; no accuracy computed. Adapt column names if an official submission template is supplied.'}
    (output / 'attachment3_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
