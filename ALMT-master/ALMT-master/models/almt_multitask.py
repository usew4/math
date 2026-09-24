"""Two heads on one ALMT fused representation: sentiment score and three classes."""

from torch import nn

from .almt import ALMT


class ALMTMultiTask(ALMT):
    def __init__(self, args):
        if int(args.model.output_dim) != 1:
            raise ValueError("The score head must have output_dim=1")
        super().__init__(args)
        self.classification_layer = nn.Linear(args.model.token_dim, 3)

    def forward(self, x_visual, x_audio, x_text):
        features = self.forward_features(x_visual, x_audio, x_text)
        return self.regression_layer(features), self.classification_layer(features)
