import runtime
import torch
from models.q2_reliability import Q2ReliabilityModel
from data import masks_and_domain


class RobustExperts(Q2ReliabilityModel):
    """Original dual-expert weights with safe handling of local missing slots."""
    def __init__(self, options):
        super().__init__({'model': options}, torch.zeros(74), torch.ones(74), torch.zeros(35), torch.ones(35))

    def forward(self, vision, audio, text):
        token_mask, content, am, vm, _ = masks_and_domain(vision, audio, text)
        safe_text = text.clone()
        safe_text[:, 1] = token_mask.long()
        empty = ~token_mask.any(1)
        safe_text[empty, 0, 0] = 101
        safe_text[empty, 1, 0] = 1
        states = self.bertmodel(safe_text)
        mean = (states * token_mask.unsqueeze(-1)).sum(1) / token_mask.sum(1, keepdim=True).clamp_min(1)
        ht = self.text_projection(torch.cat([states[:, 0], mean], -1))
        # A fully missing text sequence must not contribute learned BERT biases.
        ht = ht * content.any(1, keepdim=True)
        ha = self.audio_encoder(self._normalize(audio, am, self.audio_mean, self.audio_std), am, ht)
        hv = self.vision_encoder(self._normalize(vision, vm, self.vision_mean, self.vision_std), vm)
        ar, vr = am.float().mean(1, keepdim=True), vm.float().mean(1, keepdim=True)
        ga = self.audio_gate(torch.cat([ht, ha, ar], -1)).sigmoid() * am.any(1, keepdim=True)
        gv = self.vision_gate(torch.cat([ht, hv, vr], -1)).sigmoid() * vm.any(1, keepdim=True)
        gt = content.any(1, keepdim=True).float()
        fused = self.dropout(self.fusion_norm(gt * ht + ga * ha + gv * hv))
        full_logits = self.classification_head(fused)
        text_logits = self.text_classification_head(ht)
        trust = self.decision_gate(torch.cat([ht, fused, ar, vr, ga, gv], -1)).sigmoid()
        # When text is unavailable, route to the multimodal expert.
        trust = torch.where(content.any(1, keepdim=True), trust, torch.ones_like(trust))
        logits = trust * full_logits + (1 - trust) * text_logits
        score = 3 * torch.tanh(self.score_head(fused).squeeze(-1) / 3)
        return logits, score, text_logits, full_logits


def load_model(device):
    path = runtime.ROOT / 'checkpoints' / 'dynamic_experts_base.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    options = dict(saved.get('model_options', saved.get('config', {}).get('model', {})))
    options['bert_pretrained'] = str(runtime.ROOT / 'pretrained' / 'bert')
    model = RobustExperts(options)
    missing, unexpected = model.load_state_dict(saved['state_dict'], strict=False)
    if unexpected or missing:
        raise ValueError(f'Checkpoint mismatch: {missing}, {unexpected}')
    return model.to(device), options
