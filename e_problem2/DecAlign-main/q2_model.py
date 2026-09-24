"""DecAlign with the competition's three-class and intensity outputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models import DecAlign


class Q2DecAlign(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.head_type = getattr(args, "head_type", "flat")
        self.decalign = DecAlign(args)
        feature_dim = 6 * self.decalign.d_model
        if self.head_type == "hierarchical":
            # The DecAlign backbone and its alignment losses stay unchanged.
            self.decalign.out_layer = nn.Identity()
            self.neutral_head = nn.Linear(feature_dim, 1)
            self.polarity_head = nn.Linear(feature_dim, 2)
        elif self.head_type != "flat":
            raise ValueError(f"Unsupported head_type: {self.head_type}")
        self.intensity_head = nn.Linear(feature_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs = self.decalign(
            batch["text"], batch["audio"], batch["vision"],
            masks=(batch["text_mask"], batch["audio_mask"], batch["vision_mask"]),
        )
        if self.head_type == "hierarchical":
            # Match the flat head's final-representation dropout during training.
            features = self.decalign.out_dropout(outputs["final_rep"])
            neutral_logit = self.neutral_head(features).squeeze(-1)
            polarity_logits = self.polarity_head(features)
            polarity_log_probs = F.log_softmax(polarity_logits, dim=-1)
            nonneutral_log_prob = F.logsigmoid(-neutral_logit)
            outputs["output_logit"] = torch.stack((
                nonneutral_log_prob + polarity_log_probs[:, 0],
                F.logsigmoid(neutral_logit),
                nonneutral_log_prob + polarity_log_probs[:, 1],
            ), dim=-1)
            outputs["neutral_logit"] = neutral_logit
            outputs["polarity_logits"] = polarity_logits
        outputs["intensity"] = self.intensity_head(outputs["final_rep"]).squeeze(-1).tanh() * 3
        return outputs
