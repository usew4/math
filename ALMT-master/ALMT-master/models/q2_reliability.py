"""Text-anchored, mask-aware multimodal model for polarity and intensity."""
from __future__ import annotations

import torch
from torch import nn

from .bert import BertTextEncoder


class MaskedModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float,
                 text_guided: bool = False, temporal_conv: bool = False,
                 temporal_mamba: bool = False, mamba_seed: int = 1042):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.temporal_mamba = temporal_mamba
        if temporal_mamba:
            try:
                from mamba_ssm import Mamba
            except ImportError as error:
                raise ImportError(
                    "This experiment requires mamba_ssm; select the installed Mamba Python environment"
                ) from error
            # Preserve the initialization of every pre-existing model component.
            with torch.random.fork_rng():
                torch.manual_seed(mamba_seed)
                self.mamba = Mamba(d_model=hidden_dim, d_state=16, d_conv=4, expand=2)
            self.mamba_norm = nn.LayerNorm(hidden_dim)
            self.mamba_scale = nn.Parameter(torch.tensor(0.1))
        self.temporal_conv = temporal_conv
        if temporal_conv:
            self.temporal_depthwise = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1,
                groups=hidden_dim, bias=False)
            self.temporal_pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.temporal_activation = nn.GELU()
            self.temporal_dropout = nn.Dropout(dropout)
            self.temporal_scale = nn.Parameter(torch.tensor(0.1))
        self.text_guided = text_guided
        if text_guided:
            self.frame_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.text_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.guided_attention = nn.Linear(hidden_dim, 1, bias=False)
            # Start from the original pooling rule; learn the text correction.
            nn.init.zeros_(self.guided_attention.weight)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                text_feature: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.projection(values)
        if self.temporal_mamba:
            observed = encoded * mask.unsqueeze(-1)
            contextual = self.mamba(observed.contiguous())
            encoded = self.mamba_norm(
                encoded + torch.tanh(self.mamba_scale) * contextual)
            encoded = encoded * mask.unsqueeze(-1)
        if self.temporal_conv:
            # Exclude padded positions from the convolution and the pooled result.
            encoded = encoded * mask.unsqueeze(-1)
            local = self.temporal_depthwise(encoded.transpose(1, 2))
            local = self.temporal_pointwise(self.temporal_activation(local))
            local = self.temporal_dropout(local.transpose(1, 2))
            encoded = (encoded + torch.tanh(self.temporal_scale) * local) * mask.unsqueeze(-1)
        logits = self.attention(encoded).squeeze(-1)
        if self.text_guided:
            if text_feature is None:
                raise ValueError("Text-guided attention requires a text feature")
            joint = torch.tanh(self.frame_query(encoded) +
                               self.text_query(text_feature).unsqueeze(1))
            logits = logits + self.guided_attention(joint).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = logits.softmax(dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (encoded * weights.unsqueeze(-1)).sum(dim=1)


class WordAlignedMAG(nn.Module):
    """Bounded word-level acoustic/visual shift of BERT embeddings."""

    def __init__(self, hidden_dim: int, beta: float):
        super().__init__()
        if beta <= 0:
            raise ValueError("mag_beta must be positive")
        self.beta = beta
        self.audio_gate = nn.Linear(hidden_dim + 74, 1)
        self.vision_gate = nn.Linear(hidden_dim + 35, 1)
        self.audio_shift = nn.Linear(74, hidden_dim, bias=False)
        self.vision_shift = nn.Linear(35, hidden_dim, bias=False)
        nn.init.constant_(self.audio_gate.bias, -2.0)
        nn.init.constant_(self.vision_gate.bias, -2.0)

    def forward(self, words: torch.Tensor, audio: torch.Tensor,
                vision: torch.Tensor, audio_mask: torch.Tensor,
                vision_mask: torch.Tensor) -> torch.Tensor:
        audio_gate = torch.sigmoid(self.audio_gate(torch.cat((words, audio), dim=-1)))
        vision_gate = torch.sigmoid(self.vision_gate(torch.cat((words, vision), dim=-1)))
        shift = (audio_gate * self.audio_shift(audio) * audio_mask.unsqueeze(-1) +
                 vision_gate * self.vision_shift(vision) * vision_mask.unsqueeze(-1))
        # MAG limits the nonverbal shift relative to the lexical vector norm.
        word_norm = words.float().norm(dim=-1, keepdim=True)
        shift_norm = shift.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.beta * word_norm / shift_norm).clamp(max=1.0)
        return words + scale * shift


class Q2ReliabilityModel(nn.Module):
    def __init__(self, config: dict, audio_mean: torch.Tensor, audio_std: torch.Tensor,
                 vision_mean: torch.Tensor, vision_std: torch.Tensor):
        super().__init__()
        options = config["model"]
        dim = int(options["hidden_dim"])
        dropout = float(options["dropout"])
        self.bertmodel = BertTextEncoder(
            use_finetune=True, transformers="bert", pretrained=options["bert_pretrained"])
        frozen_layers = int(options["freeze_bert_layers"])
        if not 0 <= frozen_layers <= 12:
            raise ValueError("freeze_bert_layers must be between 0 and 12")
        for parameter in self.bertmodel.model.embeddings.parameters():
            parameter.requires_grad = False
        for layer in self.bertmodel.model.encoder.layer[:frozen_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad = False
        if self.bertmodel.model.pooler is not None:
            for parameter in self.bertmodel.model.pooler.parameters():
                parameter.requires_grad = False
        self.text_projection = nn.Sequential(
            nn.Linear(1536, dim), nn.GELU(), nn.LayerNorm(dim), nn.Dropout(dropout))
        self.mag = (WordAlignedMAG(768, float(options.get("mag_beta", 0.2)))
                    if options.get("word_aligned_mag", False) else None)
        self.audio_encoder = MaskedModalityEncoder(
            74, dim, dropout,
            text_guided=bool(options.get("audio_text_guided", False)),
            temporal_conv=bool(options.get("audio_temporal_conv", False)),
            temporal_mamba=bool(options.get("audio_temporal_mamba", False)))
        self.vision_encoder = MaskedModalityEncoder(35, dim, dropout)
        self.audio_gate = nn.Linear(2 * dim + 1, 1)
        self.vision_gate = nn.Linear(2 * dim + 1, 1)
        nn.init.constant_(self.audio_gate.bias, -2.0)
        nn.init.constant_(self.vision_gate.bias, -2.0)
        self.fusion_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.classification_head = nn.Linear(dim, 3)
        self.score_head = nn.Linear(dim, 1)
        self.neutral_head = (nn.Linear(dim, 1)
                             if options.get("neutral_auxiliary_head", False) else None)
        self.decision_fusion = bool(options.get("decision_fusion", False))
        if self.decision_fusion and self.neutral_head is not None:
            raise ValueError("Use decision fusion and neutral auxiliary head in separate experiments")
        if self.decision_fusion:
            self.text_classification_head = nn.Linear(dim, 3)
            self.decision_gate = nn.Sequential(
                nn.Linear(2 * dim + 4, dim // 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(dim // 2, 1))
            with torch.no_grad():
                self.text_classification_head.weight.copy_(self.classification_head.weight)
                self.text_classification_head.bias.copy_(self.classification_head.bias)
            nn.init.zeros_(self.decision_gate[-1].weight)
            nn.init.constant_(self.decision_gate[-1].bias, 1.5)
        self.register_buffer("audio_mean", audio_mean.float().reshape(1, 1, 74))
        self.register_buffer("audio_std", audio_std.float().reshape(1, 1, 74).clamp_min(1e-3))
        self.register_buffer("vision_mean", vision_mean.float().reshape(1, 1, 35))
        self.register_buffer("vision_std", vision_std.float().reshape(1, 1, 35).clamp_min(1e-3))

    def _normalize(self, values: torch.Tensor, mask: torch.Tensor,
                   mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return ((values - mean) / std).clamp(-5, 5) * mask.unsqueeze(-1)

    def forward(self, vision: torch.Tensor, audio: torch.Tensor,
                text: torch.Tensor) -> tuple[torch.Tensor, ...]:
        token_mask = text[:, 1, :].bool()
        audio_mask = audio.abs().sum(dim=-1) > 0
        vision_mask = vision.abs().sum(dim=-1) > 0
        audio_values = self._normalize(audio, audio_mask, self.audio_mean, self.audio_std)
        vision_values = self._normalize(vision, vision_mask, self.vision_mean, self.vision_std)
        if self.mag is None:
            token_states = self.bertmodel(text)
        else:
            bert = self.bertmodel.model
            input_ids = text[:, 0, :].long()
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            word_positions = (positions > 0) & (positions < token_mask.sum(dim=1, keepdim=True) - 1)
            word_embeddings = bert.embeddings(
                input_ids=input_ids, token_type_ids=text[:, 2, :].long())
            adapted = self.mag(word_embeddings, audio_values, vision_values,
                               audio_mask & word_positions,
                               vision_mask & word_positions)
            attention_mask = bert.get_extended_attention_mask(
                token_mask.long(), input_ids.shape)
            token_states = bert.encoder(
                adapted, attention_mask=attention_mask,
                return_dict=True).last_hidden_state
        token_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        text_mean = (token_states * token_mask.unsqueeze(-1)).sum(dim=1) / token_count
        text_feature = self.text_projection(torch.cat((token_states[:, 0], text_mean), dim=-1))

        audio_feature = self.audio_encoder(
            audio_values, audio_mask, text_feature)
        vision_feature = self.vision_encoder(
            vision_values, vision_mask)
        audio_ratio = audio_mask.float().mean(dim=1, keepdim=True)
        vision_ratio = vision_mask.float().mean(dim=1, keepdim=True)
        audio_gate = torch.sigmoid(self.audio_gate(
            torch.cat((text_feature, audio_feature, audio_ratio), dim=-1)))
        vision_gate = torch.sigmoid(self.vision_gate(
            torch.cat((text_feature, vision_feature, vision_ratio), dim=-1)))
        audio_gate = audio_gate * audio_mask.any(dim=1, keepdim=True)
        vision_gate = vision_gate * vision_mask.any(dim=1, keepdim=True)
        fused = self.dropout(self.fusion_norm(
            text_feature + audio_gate * audio_feature + vision_gate * vision_feature))
        full_logits = self.classification_head(fused)
        if self.decision_fusion:
            text_logits = self.text_classification_head(text_feature)
            trust = torch.sigmoid(self.decision_gate(torch.cat((
                text_feature, fused, audio_ratio, vision_ratio,
                audio_gate, vision_gate), dim=-1)))
            logits = trust * full_logits + (1.0 - trust) * text_logits
        else:
            logits = full_logits
        score = 3.0 * torch.tanh(self.score_head(fused).squeeze(-1) / 3.0)
        if self.decision_fusion:
            return logits, score, text_logits, full_logits
        if self.neutral_head is not None:
            return logits, score, self.neutral_head(fused).squeeze(-1)
        return logits, score
