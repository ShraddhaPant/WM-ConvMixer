"""
sca.py

H3: Structure-Conditioned Attention / Morphology-Guided Gating (SCA).

Corrected design (paper Sec. VI, Eq. 1-3, Fig. 2):

    raw EL image
         |
         +----------------------+
         |                      |
         v                      v
   Directional morphology    Haar DWT (from wavelet_patchify.HaarDWT2D)
         |                      |
         v                      v
       Mmorph                Mwavelet
         |                      |
         +----------+-----------+
                     v
              Fuse -> Mjoint      (computed ONCE per image, not per block)
                     |
    F (ConvMixer feature map, any block)
         |            |
         v            v
    Conv1x1(F)  +  lambda * Mjoint
                     |
                     v
                sigmoid -> A
                     |
                     v
           F_SCA = F (*) A + F

Mmorph is now computed on the RAW IMAGE (independent structural evidence),
not on a learned map derived from F — this matches the paper's framing of
morphology as a fixed prior, and lets Mjoint be reused across every block
instead of recomputed 8 times.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Fixed directional structuring elements
# --------------------------------------------------------------------------- #
def build_line_offsets(length: int):
    """
    Offsets for four fixed linear structuring elements at 0/45/90/135 deg.
    Returns a list of four lists of (dy, dx) offsets.
    """
    if length < 3:
        raise ValueError("Structuring-element length must be >= 3.")
    if length % 2 == 0:
        raise ValueError("Structuring-element length must be odd.")

    radius = length // 2
    offsets_0, offsets_45, offsets_90, offsets_135 = [], [], [], []

    for t in range(-radius, radius + 1):
        offsets_0.append((0, t))        # 0 deg: horizontal
        offsets_45.append((-t, t))      # 45 deg: diagonal /
        offsets_90.append((t, 0))       # 90 deg: vertical
        offsets_135.append((t, t))      # 135 deg: diagonal \

    return [offsets_0, offsets_45, offsets_90, offsets_135]


def normalize_per_image(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image min-max normalization to [0, 1]. Shared by Mmorph and Mwavelet."""
    B = M.shape[0]
    flat = M.reshape(B, -1)
    m_min = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    m_max = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
    return (M - m_min) / (m_max - m_min + eps)


# --------------------------------------------------------------------------- #
# Directional morphology -> Mmorph (Sec VI-C)
# --------------------------------------------------------------------------- #
class DirectionalMorphology(nn.Module):
    """
    Fixed directional morphological gradient: dilation - erosion at
    0/45/90/135 deg, elementwise-max reduced, reflect-padded, normalized
    per image. Operates on a single-channel map (typically the raw
    grayscale EL image).

    Default length=5 (was 7): a 5-pixel line is a smaller fraction of the
    downstream 30x30 ConvMixer feature grid, preserving finer structure.
    """

    def __init__(self, length: int = 5):
        super().__init__()
        self.length = length
        self.radius = length // 2
        offsets = build_line_offsets(length)
        self.offsets = tuple(tuple(direction) for direction in offsets)

    def shift(self, x: torch.Tensor, dy: int, dx: int):
        r = self.radius
        padded = F.pad(x, (r, r, r, r), mode="reflect")
        H, W = x.shape[-2:]
        y_start = r + dy
        x_start = r + dx
        return padded[:, :, y_start:y_start + H, x_start:x_start + W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DirectionalMorphology expects (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        directional_gradients = []
        for direction in self.offsets:
            samples = torch.stack([self.shift(x, dy, dx) for dy, dx in direction], dim=0)
            dilation = samples.amax(dim=0)
            erosion = samples.amin(dim=0)
            directional_gradients.append(dilation - erosion)

        M = torch.stack(directional_gradients, dim=0).amax(dim=0)
        return normalize_per_image(M)


# --------------------------------------------------------------------------- #
# Fuse(Mwavelet, Mmorph) -> Mjoint (Fig. 2 / Eq. 2)
# --------------------------------------------------------------------------- #
class GuidanceFusion(nn.Module):
    """mode in {"conv", "max", "mean"}."""

    def __init__(self, mode: str = "conv"):
        super().__init__()
        self.mode = mode
        if mode == "conv":
            self.fuse = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, Mwavelet: torch.Tensor, Mmorph: torch.Tensor) -> torch.Tensor:
        if Mwavelet.shape[-2:] != Mmorph.shape[-2:]:
            Mmorph = F.interpolate(Mmorph, size=Mwavelet.shape[-2:], mode="bilinear", align_corners=False)
        if self.mode == "conv":
            return self.fuse(torch.cat([Mwavelet, Mmorph], dim=1))
        if self.mode == "max":
            return torch.maximum(Mwavelet, Mmorph)
        if self.mode == "mean":
            return (Mwavelet + Mmorph) / 2.0
        raise ValueError(f"Unknown fusion mode: {self.mode}")


# --------------------------------------------------------------------------- #
# SCA gate: Eq. 1 and 3 (lightweight — Mjoint computed upstream, once)
# --------------------------------------------------------------------------- #
class SCA(nn.Module):
    """
    A = sigmoid(Conv1x1(F) + lambda * Mjoint)   (Eq. 1)
    F_SCA = F * A + F                            (Eq. 3)

    Mjoint is passed in (computed once per image by the caller), not
    recomputed here — this is what makes SCA cheap to place at every block.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.feature_gate = nn.Conv2d(dim, 1, kernel_size=1)
        self.lambda_m = nn.Parameter(torch.tensor(0.0))

    def forward(self, F_features: torch.Tensor, Mjoint: torch.Tensor) -> torch.Tensor:
        if F_features.ndim != 4:
            raise ValueError(f"SCA expects (B,C,H,W), got {tuple(F_features.shape)}")
        if Mjoint.shape[-2:] != F_features.shape[-2:]:
            Mjoint = F.interpolate(Mjoint, size=F_features.shape[-2:], mode="bilinear", align_corners=False)
        A = torch.sigmoid(self.feature_gate(F_features) + self.lambda_m * Mjoint)
        return F_features * A + F_features


# --------------------------------------------------------------------------- #
# Standalone test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(42)

    raw_img = torch.randn(2, 1, 300, 300)
    feat = torch.randn(2, 128, 30, 30, requires_grad=True)

    morphology = DirectionalMorphology(length=5)
    Mmorph = morphology(raw_img)
    print("Mmorph shape:", tuple(Mmorph.shape), "range",
          f"[{Mmorph.min().item():.4f}, {Mmorph.max().item():.4f}]")

    Mwavelet = torch.rand(2, 1, 150, 150)  # placeholder for the real DWT-derived map
    fusion = GuidanceFusion(mode="conv")
    Mjoint = fusion(Mwavelet, Mmorph)
    print("Mjoint shape:", tuple(Mjoint.shape))

    sca = SCA(dim=128)
    out = sca(feat, Mjoint)
    print("SCA output shape:", tuple(out.shape))
    assert out.shape == feat.shape

    loss = out.mean()
    loss.backward()
    assert feat.grad is not None
    assert sca.lambda_m.grad is not None
    print("Gradient test: PASSED")