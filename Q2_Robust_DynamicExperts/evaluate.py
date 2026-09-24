"""Click Run: evaluate the retained dual-expert model under fixed local gaps."""
import argparse
import csv
import json
import runtime
import numpy as np
from data import load_data, PATTERNS
from model import load_model
from engine import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--split', choices=['valid', 'test'], default='valid')
    args = parser.parse_args()
    config = runtime.settings(args.config)
    device, dtype, amp = runtime.environment(config)
    output = runtime.output_directory()
    dataset = load_data(config['data'])[args.split]
    conditions = [None]
    for seed in config['validation']['suite_seeds']:
        for pattern in PATTERNS:
            for rate in config['missing']['rates']:
                conditions.append({'pattern': pattern, 'rate': rate, 'fixed_seed': seed})
        for pattern in PATTERNS:
            for position in ('front', 'middle', 'back'):
                conditions.append({'pattern': pattern, 'rate': .3, 'fixed_seed': seed, 'position': position})
        for pattern in ('TA', 'TV', 'AV', 'TAV'):
            conditions.append({'pattern': pattern, 'rate': .3, 'fixed_seed': seed, 'synchronous': True})
    records = []
    model, _ = load_model(device)
    for i, condition in enumerate(conditions):
        result, rows = evaluate(model, dataset, config, device, dtype, amp, condition)
        spec = condition or {'pattern': 'clean', 'rate': 0, 'fixed_seed': 0}
        records.append({'model': 'baseline', **spec, 'position': spec.get('position', 'random'),
                        'synchronous': spec.get('synchronous', False), **result})
        if condition is None:
            with (output / f'{args.split}_clean_predictions.csv').open('w', newline='', encoding='utf-8-sig') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f'{i + 1}/{len(conditions)} {spec} F1={result["macro_f1"]:.4f}', flush=True)
    path = output / f'{args.split}_robustness.json'
    path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    fields = ['model', 'pattern', 'rate', 'position', 'synchronous', 'fixed_seed', 'accuracy', 'macro_f1', 'mae', 'pearson']
    with (output / f'{args.split}_robustness.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(records)
    grouped = {}
    for r in records:
        key = (r['model'], r['pattern'], r['rate'], r['position'], r['synchronous'])
        grouped.setdefault(key, []).append(r)
    summary = []
    for key, group in grouped.items():
        item = dict(zip(['model', 'pattern', 'rate', 'position', 'synchronous'], key))
        for metric in ('accuracy', 'macro_f1', 'mae', 'pearson'):
            values = [r[metric] for r in group if r[metric] is not None]
            item[metric + '_mean'] = float(np.mean(values)) if values else None
            item[metric + '_std'] = float(np.std(values)) if values else None
        item['mask_repeats'] = len(group)
        summary.append(item)
    (output / f'{args.split}_robustness_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    from plots import render
    render(records, output, args.split)
    print(f'Saved {path}', flush=True)


if __name__ == '__main__': main()
