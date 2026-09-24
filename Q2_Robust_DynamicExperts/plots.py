"""Standard exportable figures for the robustness report."""
import numpy as np


def render(records, output, split):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('Install matplotlib to render figures; all numerical results have been saved.')
        return
    from data import PATTERNS
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    for ax, pattern in zip(axes.flat, PATTERNS):
        clean = next(r['macro_f1'] for r in records if r['pattern'] == 'clean')
        relevant = [r for r in records if r['pattern'] == pattern and r['position'] == 'random' and not r['synchronous']]
        rates = sorted(set(r['rate'] for r in relevant))
        values = [[r['macro_f1'] for r in relevant if r['rate'] == rate] for rate in rates]
        ax.errorbar([0] + rates, [clean] + [np.mean(v) for v in values], yerr=[0] + [np.std(v) for v in values], marker='o', color='#1f77b4')
        ax.set_title(f'Missing: {pattern}'); ax.grid(alpha=.2); ax.set_xlabel('Target missing span / valid length'); ax.set_ylabel('Macro-F1')
    axes.flat[-1].axis('off')
    fig.tight_layout(); fig.savefig(output / f'{split}_missing_rate_f1.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.array(next(r['confusion'] for r in records if r['pattern'] == 'clean'))
    ax.imshow(matrix, cmap='Blues')
    for i in range(3):
        for j in range(3): ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
    ax.set_xticks(range(3), ['Neg', 'Neu', 'Pos']); ax.set_yticks(range(3), ['Neg', 'Neu', 'Pos'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Original input')
    fig.tight_layout(); fig.savefig(output / f'{split}_clean_confusion.png', dpi=180); plt.close(fig)
