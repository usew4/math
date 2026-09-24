"""Text-anchored cross-modal attention with masks for unaligned streams."""
from __future__ import annotations

import torch
from torch import nn


class UnalignedFusion(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4, dropout: float = .15,
                 auxiliary_heads: bool = False):
        super().__init__()
        self.auxiliary_heads = auxiliary_heads
        self.text_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, width))
        self.audio_proj = nn.Sequential(nn.LayerNorm(74), nn.Linear(74, width))
        self.vision_proj = nn.Sequential(nn.LayerNorm(35), nn.Linear(35, width))
        self.cls = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.audio_null = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.vision_null = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.text_pos = nn.Parameter(torch.randn(1, 51, width) * .02)
        self.audio_pos = nn.Parameter(torch.randn(1, 101, width) * .02)
        self.vision_pos = nn.Parameter(torch.randn(1, 101, width) * .02)

        def encoder() -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=width, nhead=heads, dim_feedforward=width * 2,
                dropout=dropout, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)

        self.text_encoder = encoder()
        self.audio_encoder = encoder()
        self.vision_encoder = encoder()
        self.text_to_audio = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.text_to_vision = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(width * 3 + 3, width), nn.GELU(),
                                  nn.Linear(width, 3))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                  nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(width, 3)
        self.regressor = nn.Linear(width, 1)
        if auxiliary_heads:
            self.unimodal_classifiers = nn.ModuleDict({
                name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 3))
                for name in ("text", "audio", "vision")
            })
            self.unimodal_regressors = nn.ModuleDict({
                name: nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
                for name in ("text", "audio", "vision")
            })

    def forward(self, batch: dict[str, torch.Tensor], *, return_aux: bool = False):
        if return_aux and not self.auxiliary_heads:
            raise ValueError("This checkpoint has no unimodal prediction heads")
        text, audio, vision = batch["text"], batch["audio"], batch["vision"]
        tm, am, vm = batch["text_mask"].bool(), batch["audio_mask"].bool(), batch["vision_mask"].bool()
        b = len(text)
        t = torch.cat([self.cls.expand(b, -1, -1),
                       self.text_proj(text) * tm.unsqueeze(-1)], dim=1) + self.text_pos
        a = torch.cat([self.audio_null.expand(b, -1, -1),
                       self.audio_proj(audio) * am.unsqueeze(-1)], dim=1) + self.audio_pos
        v = torch.cat([self.vision_null.expand(b, -1, -1),
                       self.vision_proj(vision) * vm.unsqueeze(-1)], dim=1) + self.vision_pos
        tm = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=tm.device), tm], dim=1)
        am = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=am.device), am], dim=1)
        vm = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=vm.device), vm], dim=1)
        t = self.text_encoder(t, src_key_padding_mask=~tm)
        a = self.audio_encoder(a, src_key_padding_mask=~am)
        v = self.vision_encoder(v, src_key_padding_mask=~vm)
        coverage = torch.stack([tm[:, 1:].float().mean(1), am[:, 1:].float().mean(1),
                                vm[:, 1:].float().mean(1)], dim=1)
        availability = coverage > 0
        if return_aux:
            summaries = {"text": t[:, 0], "audio": a[:, 0], "vision": v[:, 0]}
            aux_logits = torch.stack([
                self.unimodal_classifiers[name](summaries[name])
                for name in ("text", "audio", "vision")
            ], dim=1)
            aux_intensity = torch.stack([
                self.unimodal_regressors[name](summaries[name]).squeeze(-1).tanh() * 3
                for name in ("text", "audio", "vision")
            ], dim=1)
        ta, _ = self.text_to_audio(t, a, a, key_padding_mask=~am, need_weights=False)
        tv, _ = self.text_to_vision(t, v, v, key_padding_mask=~vm, need_weights=False)
        cov = coverage[:, None, :].expand(-1, t.shape[1], -1)
        gates = self.gate(torch.cat([t, ta, tv, cov], dim=-1))
        gate_availability = availability.clone()
        gate_availability[:, 0] |= ~gate_availability.any(dim=1)
        gates = gates.masked_fill(~gate_availability[:, None, :], -1e4).softmax(dim=-1)
        fused = gates[..., 0:1] * t + gates[..., 1:2] * ta + gates[..., 2:3] * tv
        out = self.head(fused[:, 0])
        logits = self.classifier(out)
        intensity = self.regressor(out).squeeze(-1).tanh() * 3
        if return_aux:
            return logits, intensity, {
                "logits": aux_logits, "intensity": aux_intensity,
                "availability": availability,
            }
        return logits, intensity
