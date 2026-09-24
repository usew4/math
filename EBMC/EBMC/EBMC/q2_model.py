"""Q2 adapter; original experiment entrypoints retain their interfaces."""
import torch
from torch import nn
from torch.nn import functional as F
from ebmc import EBMC


class Q2Model(EBMC):
    def __init__(self, args):
        super().__init__(args, 74, 768, 35, args.hidden, 1,
                         depth=args.depth, num_heads=args.num_heads,
                         drop_rate=args.drop_rate, attn_drop_rate=0.0)
        self.classifier = nn.Linear(3 * args.hidden, 3)
        # Teacher is explicitly managed, frozen, and stored in a separate file.
        object.__setattr__(self, '_q2_teacher', None)

    def set_teacher(self, teacher):
        teacher.eval()
        teacher.requires_grad_(False)
        object.__setattr__(self, '_q2_teacher', teacher)

    def _load_teacher(self, inputfeats, input_features_mask, umask):
        teacher = self._q2_teacher
        if teacher is None:
            raise RuntimeError('Stage II requires an explicitly loaded Stage-I teacher')
        teacher.eval()
        with torch.no_grad():
            out = teacher(inputfeats, input_features_mask, umask, first_stage=True)
        return out[2:5]

    def _forward_stage1(self, x_a, x_t, x_v, B, seq_len, label, umask,
                        do_cf, weight_save):
        if label is not None:
            return super()._forward_stage1(x_a, x_t, x_v, B, seq_len,
                                           label, umask, do_cf, weight_save)
        logits = [self.nlp_head_a(x_a), self.nlp_head_t(x_t), self.nlp_head_v(x_v)]
        return x_a, sum(logits) / 3, *logits, {}, weight_save

    def _forward_stage2(self, x_a, x_t, x_v, weight_a, weight_t, weight_v,
                        B, seq_len, label, umask, do_cf, weight_save,
                        inputfeats, input_features_mask):
        if label is not None:
            return super()._forward_stage2(
                x_a, x_t, x_v, weight_a, weight_t, weight_v, B, seq_len,
                label, umask, do_cf, weight_save, inputfeats, input_features_mask)
        xs = [torch.sum(w * x.reshape(B, seq_len, 3, self.D_e), dim=2)
              for x, w in zip((x_a, x_t, x_v), (weight_a, weight_t, weight_v))]
        joint = torch.cat(xs, dim=-1)
        hidden = joint + F.dropout(F.relu(self.proj1(joint)),
                                   p=self.out_dropout, training=self.training)
        return (hidden, self.nlp_head(hidden), self.nlp_head_a(xs[0]),
                self.nlp_head_t(xs[1]), self.nlp_head_v(xs[2]), {}, weight_save)

    def run_batch(self, batch, stage, with_loss=False):
        out = self(batch['features'], batch['valid'], batch['umask'],
                   first_stage=stage == 1,
                   label=batch['y'].reshape(-1, 1) if with_loss else None)
        result = {'regression': out[1].reshape(-1),
                  'unimodal': [x.reshape(-1) for x in out[2:5]], 'aux': out[5]}
        if stage == 2:
            result['classification'] = self.classifier(out[0].squeeze(1))
        return result
