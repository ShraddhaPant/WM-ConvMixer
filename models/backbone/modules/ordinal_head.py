import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    """
    CORAL-style ordinal prediction head (Cao et al., 2020),
    fulfilling review Section II-A's "ordinal prediction head" spec.

    Instead of K independent output units (nominal), this uses ONE
    shared weight vector plus (K-1) ordered thresholds. Predicting
    "is severity > Healthy?", "> Mild?", "> Moderate?" as 3 binary
    questions, with thresholds constrained non-increasing so the
    predictions can never rank-invert.
    """

    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1

        self.shared = nn.Linear(in_features, 1, bias=False)
        self.bias_0 = nn.Parameter(torch.zeros(1))
        if self.num_thresholds > 1:
            self.bias_deltas = nn.Parameter(torch.zeros(self.num_thresholds - 1))
        else:
            self.bias_deltas = None

    def forward(self, x):
        """
        Input:  x [B, in_features]
        Output: logits [B, num_thresholds] — one per ordinal cutoff.
        """
        g = self.shared(x)  # [B, 1]
        biases = [self.bias_0]
        if self.bias_deltas is not None:
            cur = self.bias_0
            for i in range(self.bias_deltas.shape[0]):
                cur = cur - F.softplus(self.bias_deltas[i])  # enforce non-increasing
                biases.append(cur)
        biases = torch.cat(biases)          # [num_thresholds]
        return g + biases.unsqueeze(0)       # [B, num_thresholds]

    @staticmethod
    def encode_labels(y, num_classes=4):
        """Integer label [B] -> extended binary targets [B, num_classes-1]."""
        thresholds = torch.arange(num_classes - 1, device=y.device).unsqueeze(0)
        return (y.unsqueeze(1) > thresholds).float()

    @staticmethod
    def predict_class(logits):
        """Logits [B, num_thresholds] -> predicted ordinal class [B]."""
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)