import torch
import torch.nn as nn


class LGAT(nn.Module):
    """
    Lightweight Global Attention Token (review Section VII-B, IX-B stage 5).

    One learned query, independent of the input image — not derived from x,
    unlike a CLS-token design. Keys and values come from all spatial tokens.
    Produces a single 1xN attention vector (not the full NxN matrix that
    full self-attention would form) — this is what removes the quadratic
    attention-score term under the review's stated formulation (Section VII-C).
    """

    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # input-independent
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, x):
        """
        Input:  x [B, C, H, W] — a spatial feature map (e.g. ConvMixer's
                output before pooling).
        Output: g [B, C] — the global descriptor.
        """
        B, C, H, W = x.shape
        Z = x.flatten(2).transpose(1, 2)              # [B, N=H*W, C]
        q = self.q_proj(self.query.expand(B, -1, -1)) # [B, 1, C]
        K = self.k_proj(Z)                             # [B, N, C]
        V = self.v_proj(Z)                              # [B, N, C]

        attn = torch.softmax((q @ K.transpose(-2, -1)) * self.scale, dim=-1)  # [B, 1, N]
        g = attn @ V                                    # [B, 1, C]
        return g.squeeze(1)                              # [B, C]