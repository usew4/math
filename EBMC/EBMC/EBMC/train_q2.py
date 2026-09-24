"""Standalone Q2 trainer. Does not load test or attachment 3 during training."""
import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from q2_data import Q2Dataset, COMBINATIONS, load_splits, collate_q2
from q2_model import Q2Model


def model_args(c, device):
    return SimpleNamespace(**c, dataset='CMUMOSEI', device=torch.device(device),
                           frame_seq=True, no_cuda=device == 'cpu', n_classes=1,
                           test_condition='atv')


def move(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def loader(ds, c, shuffle=False):
    g = torch.Generator().manual_seed(c['seed'] + ds.epoch)
    return DataLoader(ds, batch_size=c['batch_size'], shuffle=shuffle,
                      generator=g, num_workers=c['workers'], collate_fn=collate_q2)


def metrics(y, pred, cls, logits):
    corr = None if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0 else float(np.corrcoef(y, pred)[0, 1])
    out = dict(mae=float(np.abs(y - pred).mean()), corr=corr)
    if logits is not None:
        p = logits.argmax(-1)
        out.update(accuracy=float(accuracy_score(cls, p)),
                   macro_f1=float(f1_score(cls, p, labels=[0, 1, 2], average='macro', zero_division=0)),
                   class_f1=f1_score(cls, p, labels=[0, 1, 2], average=None, zero_division=0).tolist())
    return out


def scenarios(c, stress=False):
    result = [('complete', 0.0, 'random')]
    rates = c['validation_rates'] + ([0.7] if stress else [])
    positions = ['front', 'middle', 'back'] if stress else ['random']
    return result + [(m, r, p) for m in COMBINATIONS for r in rates for p in positions]


@torch.no_grad()
def evaluate(model, ds, c, stage, stress=False):
    model.eval()
    rows = []
    for scene in scenarios(c, stress):
        ds.scenario = scene
        ys, ps, cs, ls, us = [], [], [], [], []
        for b in loader(ds, c):
            b = move(b, model.device)
            o = model.run_batch(b, stage)
            ys.append(b['y'].cpu().numpy()); cs.append(b['cls'].cpu().numpy())
            ps.append(o['regression'].cpu().numpy())
            us.append(torch.stack(o['unimodal'], -1).cpu().numpy())
            if stage == 2:
                ls.append(o['classification'].cpu().numpy())
        y = np.concatenate(ys)
        row = metrics(y, np.concatenate(ps), np.concatenate(cs), np.concatenate(ls) if ls else None)
        row['unimodal_mae'] = float(np.abs(np.concatenate(us) - y[:, None]).mean())
        rows.append(dict(scenario=list(scene), **row))
    avg = {k: float(np.mean([r[k] for r in rows])) for k in ('mae', 'unimodal_mae')}
    if stage == 2:
        avg.update({k: float(np.mean([r[k] for r in rows])) for k in ('accuracy', 'macro_f1')})
    return dict(average=avg, scenarios=rows)


def train_epoch(model, ds, c, stage, optimizer, max_batches=0):
    model.train()
    totals, skipped = [], 0
    for i, b in enumerate(loader(ds, c, True)):
        if max_batches and i >= max_batches:
            break
        skipped += b['skipped']
        b = move(b, model.device)
        optimizer.zero_grad(set_to_none=True)
        o = model.run_batch(b, stage, with_loss=True)
        aux = o['aux']
        loss = c['lambda_msd'] * aux['disentangle'] + c['lambda_cce'] * aux['cfd']
        if stage == 1:
            loss = loss + sum(F.mse_loss(p, b['y']) for p in o['unimodal'])
        else:
            loss = loss + F.mse_loss(o['regression'], b['y']) + c['lambda_cls'] * F.cross_entropy(o['classification'], b['cls'])
            loss = loss + c['lambda_emc'] * aux['emc'] + c['lambda_imtd'] * aux['imtd']
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss: stage={stage}, batch={i}')
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.count_nonzero() for g in grads):
            raise FloatingPointError('Missing, zero, or nonfinite gradients')
        if model._q2_teacher is not None:
            assert not model._q2_teacher.training
            assert all(p.grad is None and not p.requires_grad for p in model._q2_teacher.parameters())
        optimizer.step()
        totals.append(loss.item())
    return dict(loss=float(np.mean(totals)), skipped_short_modalities=skipped, batches=len(totals))


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s['cuda'])


def save(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def read_checkpoint(path):
    # Only load trusted checkpoints created by this training script.
    return torch.load(path, map_location='cpu', weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'q2_config.json'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', default=str(ROOT / 'runs' / 'q2'))
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume')
    ap.add_argument('--evaluate', help='Student checkpoint; evaluation only')
    ap.add_argument('--split', choices=['valid', 'test'], default='valid')
    ap.add_argument('--stress', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    for name in ['batch_size', 'stage1_epochs', 'stage2_epochs', 'seed', 'workers', 'hidden', 'depth']:
        ap.add_argument('--' + name.replace('_', '-'), type=int)
    args = ap.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    for k in c:
        if getattr(args, k, None) is not None:
            c[k] = getattr(args, k)
    if args.smoke:
        c.update(stage1_epochs=1, stage2_epochs=1, batch_size=2, workers=0)
    if args.resume and args.evaluate:
        ap.error('--resume and --evaluate are mutually exclusive')
    ck = read_checkpoint(args.resume or args.evaluate) if args.resume or args.evaluate else None
    if ck:
        # Restoring uses saved model/training settings, not silently changed CLI defaults.
        c = ck['config']
        if bool(ck.get('smoke')) != args.smoke and args.resume:
            ap.error('Resume must preserve --smoke mode')
    if c['batch_size'] < 1 or c['workers'] < 0 or min(c['stage1_epochs'], c['stage2_epochs']) < 1:
        ap.error('Batch size and stage lengths must be positive; workers must be nonnegative')
    if c['hidden'] % c['num_heads'] or c['depth'] < 1:
        ap.error('Hidden dimension must divide evenly into attention heads; depth must be positive')
    if not 0 <= c['complete_probability'] <= 1 or not 0 < c['train_rates'][0] <= c['train_rates'][1] < 1:
        ap.error('Invalid complete probability or training missing-rate interval')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; verify PyTorch CUDA build or explicitly use --device cpu')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)
    if not ck and (outdir / 'latest.pt').exists():
        raise FileExistsError('Output already has a checkpoint: use --resume or a new output directory')
    info = dict(torch=torch.__version__, cuda=torch.version.cuda, device=args.device,
                gpu=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else None,
                config=c, data=str(Path(args.data).resolve()), smoke=args.smoke)
    if args.device.startswith('cuda'):
        # Force a device operation so driver/runtime incompatibility fails before data loading.
        torch.ones(1, device=args.device).sum().item()
    print(json.dumps(info, ensure_ascii=False), flush=True)
    (outdir / 'environment.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    parts = load_splits(args.data, (args.split,) if args.evaluate else ('train', 'valid'), 4 if args.smoke else 0)
    model = Q2Model(model_args(c, args.device)).to(args.device)
    if ck:
        model.load_state_dict(ck['model'])
    if args.evaluate:
        result = evaluate(model, Q2Dataset(parts[args.split], c), c, 2, args.stress)
        (outdir / f'evaluation_{args.split}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(result['average']); return
    train, valid = Q2Dataset(parts['train'], c, True), Q2Dataset(parts['valid'], c)
    opt = torch.optim.Adam(model.parameters(), lr=c['lr'], weight_decay=c['l2'])
    start, best_teacher, best_student = 0, float('inf'), (-float('inf'), -float('inf'))
    teacher_state = None
    if ck:
        opt.load_state_dict(ck['optimizer']); start = ck['next_epoch']
        best_teacher, best_student = ck['best_teacher'], tuple(ck['best_student'])
        teacher_path = Path(args.resume).parent / 'teacher.pt'
        teacher_state = read_checkpoint(teacher_path)['model']
        if not (outdir / 'teacher.pt').exists():
            save(outdir / 'teacher.pt', read_checkpoint(teacher_path))
    teacher = None
    if ck:
        restore_rng(ck['rng'])
    for epoch in range(start, c['stage1_epochs'] + c['stage2_epochs']):
        stage = 1 if epoch < c['stage1_epochs'] else 2
        if stage == 2 and teacher is None:
            # Constructing the frozen teacher must not alter the resumed RNG stream.
            state = rng_state()
            teacher = Q2Model(model_args(c, args.device)).to(args.device)
            teacher.load_state_dict(teacher_state)
            model.set_teacher(teacher)
            restore_rng(state)
        train.epoch = epoch
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(args.device)
        report = train_epoch(model, train, c, stage, opt, 1 if args.smoke else 0)
        evaluation = evaluate(model, valid, c, stage)
        avg = evaluation['average']
        is_best = False
        if stage == 1 and avg['unimodal_mae'] < best_teacher:
            best_teacher = avg['unimodal_mae']
            teacher_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save(outdir / 'teacher.pt', dict(model=teacher_state, config=c, epoch=epoch, score=best_teacher))
        if stage == 2:
            score = (avg['macro_f1'], -avg['mae'])
            if score > best_student:
                best_student, is_best = score, True
        peak = torch.cuda.max_memory_allocated(args.device) / 2**20 if args.device.startswith('cuda') else None
        report.update(epoch=epoch, stage=stage, validation=evaluation, peak_gpu_mib=peak)
        with (outdir / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(report, allow_nan=False) + '\n')
        payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), config=c,
                       next_epoch=epoch + 1, best_teacher=best_teacher, best_student=best_student,
                       rng=rng_state(), smoke=args.smoke)
        save(outdir / 'latest.pt', payload)
        if is_best:
            save(outdir / 'best.pt', payload)
        print(f'epoch={epoch} stage={stage} train={report["loss"]:.5f} valid={avg} peak_MiB={peak}', flush=True)


if __name__ == '__main__':
    main()
