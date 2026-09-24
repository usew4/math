"""Run directly: python EBMC/test_q2.py (no extra test framework)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from q2_data import Q2Dataset, collate_q2, erase_interval, MODALITIES, COMBINATIONS
from train_q2 import model_args, move, metrics, rng_state, restore_rng, save, read_checkpoint, train_epoch
from q2_model import Q2Model


class Q2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1111)
        self.c = json.loads((Path(__file__).resolve().parents[1] / 'q2_config.json').read_text())
        self.c.update(hidden=8, depth=1, batch_size=1)
        self.d = {k: np.ones((2, 50 if k == 'text' else 500, dim), np.float32)
                  for k, _, dim in MODALITIES}
        self.d.update(lengths={'text': np.array([12, 50]), 'audio': np.array([20, 1]),
                               'vision': np.array([15, 0])}, y=np.array([-1., 1.], np.float32),
                      cls=np.array([0, 2]), ids=['a', 'b'])

    def test_contiguous_masks_and_reproducibility(self):
        ds = Q2Dataset(self.d, self.c)
        for combo in COMBINATIONS:
            ds.scenario = (combo, .3, 'random')
            x = ds[0]
            np.testing.assert_array_equal(x['features'], ds[0]['features'])
            for j, (key, letter, _) in enumerate(MODALITIES):
                missing = np.flatnonzero(x['valid'][:, j] - x['observed'][:, j])
                if letter in combo:
                    self.assertGreater(len(missing), 0)
                    self.assertTrue(np.all(np.diff(missing) == 1))
                else:
                    self.assertEqual(len(missing), 0)
                L = self.d['lengths'][key][0]
                self.assertTrue(np.all(x['observed'][L:, j] == 0))
            self.assertEqual(x['observed'][0, 1], 1)
            self.assertEqual(x['observed'][11, 1], 1)
        ds.scenario = ('TAV', .5, 'back')
        self.assertEqual(ds[1]['skipped'], 2)
        self.assertEqual(ds[1]['observed'][49, 1], 1)
        train = Q2Dataset(self.d, self.c, True)
        train.epoch = 3
        a = train[0]
        train[1]
        np.testing.assert_array_equal(a['observed'], train[0]['observed'])
        self.assertEqual(erase_interval(100, False, .3, np.random.default_rng(0), 'front'), (0, 30))
        self.assertEqual(erase_interval(100, False, .3, np.random.default_rng(0), 'back'), (70, 100))

    def test_both_stages_single_batch_and_reload(self):
        ds = Q2Dataset(self.d, self.c)
        ds.scenario = ('TAV', .5, 'middle')
        b = collate_q2([ds[0]])
        model = Q2Model(model_args(self.c, 'cpu'))
        teacher = copy.deepcopy(model)
        model.set_teacher(teacher)
        model.train()
        for stage in [1, 2]:
            model.zero_grad(set_to_none=True)
            o = model.run_batch(b, stage, with_loss=True)
            loss = sum(o['aux'].values()) + sum(x.square().mean() for x in o['unimodal'])
            if stage == 2:
                loss = loss + o['classification'].square().mean() + o['regression'].square().mean()
                self.assertEqual(o['classification'].shape, (1, 3))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertFalse(teacher.training)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        self.assertFalse(any('teacher' in k for k in model.state_dict()))
        model.eval()
        with torch.no_grad():
            expected = model.run_batch(b, 2)['classification']
            for key in ('features', 'valid', 'umask'):
                bz = dict(b); bz[key] = torch.zeros_like(b[key])
                self.assertTrue(torch.isfinite(model.run_batch(bz, 2)['regression']).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.pt'
            save(path, {'model': model.state_dict()})
            restored = Q2Model(model_args(self.c, 'cpu'))
            restored.load_state_dict(read_checkpoint(path)['model'])
            restored.eval()  # No teacher or labels needed for inference.
            b.pop('y'); b.pop('cls')
            with torch.no_grad():
                torch.testing.assert_close(expected, restored.run_batch(b, 2)['classification'], rtol=0, atol=0)

    def test_rng_and_undefined_correlation(self):
        s = rng_state()
        a, b = torch.rand(3), np.random.rand(3)
        restore_rng(s)
        torch.testing.assert_close(a, torch.rand(3))
        np.testing.assert_array_equal(b, np.random.rand(3))
        self.assertIsNone(metrics(np.ones(2), np.ones(2), None, None)['corr'])

    def test_resume_matches_next_training_update(self):
        model = Q2Model(model_args(self.c, 'cpu'))
        ds = Q2Dataset(self.d, self.c, True)
        opt = torch.optim.Adam(model.parameters(), lr=self.c['lr'])
        train_epoch(model, ds, self.c, 1, opt, max_batches=1)
        state = copy.deepcopy(dict(model=model.state_dict(), opt=opt.state_dict(), rng=rng_state()))
        ds.epoch = 1
        train_epoch(model, ds, self.c, 1, opt, max_batches=1)
        other = Q2Model(model_args(self.c, 'cpu'))
        other.load_state_dict(state['model'])
        opt2 = torch.optim.Adam(other.parameters(), lr=self.c['lr'])
        opt2.load_state_dict(state['opt'])
        restore_rng(state['rng'])
        train_epoch(other, ds, self.c, 1, opt2, max_batches=1)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, other.state_dict()[k], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
