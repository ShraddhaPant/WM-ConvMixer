"""
wm_convmixer.py

H2 + H3 assembly, corrected: Mjoint (wavelet + morphology guidance) is
computed ONCE per forward pass from the raw image, then reused across
every ConvMixer block's SCA gate — not recomputed per block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.modules.wavelet_patchify import WaveletPatchify, HaarDWT2D
from models.backbone.modules.sca import DirectionalMorphology, GuidanceFusion, SCA, normalize_per_image
from models.backbone.convmixer import ConvMixerBlock


class WMConvMixer(nn.Module):
    def __init__(
        self,
        in_channels=1,
        dim=128,
        depth=8,
        kernel_size=9,
        wavelet_patch_size=5,
        morphology_length=5,
        num_classes=4,
        use_sca=True,
        gate_layers=None,      # e.g. [5,6,7] to gate only the last 3 blocks; None = all
        fusion_mode="conv",
    ):
        super().__init__()

        self.config = dict(
            dim=dim, depth=depth, kernel_size=kernel_size,
            wavelet_patch_size=wavelet_patch_size, morphology_length=morphology_length,
            in_channels=in_channels, use_sca=use_sca, gate_layers=gate_layers,
        )

        self.wavelet_patchify = WaveletPatchify(
            in_channels=in_channels, embed_dim=dim, patch_size=wavelet_patch_size,
        )
        self.blocks = nn.ModuleList(
            [ConvMixerBlock(dim, kernel_size) for _ in range(depth)]
        )

        self.use_sca = use_sca
        self.gate_layers = set(gate_layers) if gate_layers is not None else set(range(depth))

        if use_sca:
            self.morphology = DirectionalMorphology(length=morphology_length)
            self.wavelet_dwt = HaarDWT2D()  # separate small pass, guidance only
            self.fusion = GuidanceFusion(mode=fusion_mode)
            self.gates = nn.ModuleDict(
                {str(i): SCA(dim) for i in self.gate_layers}
            )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def _compute_guidance(self, x: torch.Tensor) -> torch.Tensor:
        """Mjoint = Fuse(Mwavelet, Mmorph), computed once from the raw image."""
        Mmorph = self.morphology(x)  # (B,1,H,W), normalized

        B, C, H, W = x.shape
        dwt_out = self.wavelet_dwt(x)                          # (B,4C,H/2,W/2)
        dwt_out = dwt_out.view(B, C, 4, dwt_out.shape[-2], dwt_out.shape[-1])
        LH, HL, HH = dwt_out[:, :, 1], dwt_out[:, :, 2], dwt_out[:, :, 3]
        mag = torch.sqrt(LH.pow(2) + HL.pow(2) + HH.pow(2) + 1e-6).mean(dim=1, keepdim=True)
        Mwavelet = normalize_per_image(mag)

        return self.fusion(Mwavelet, Mmorph)

    def forward(self, x: torch.Tensor):
        Mjoint = self._compute_guidance(x) if self.use_sca else None

        tokens, spatial_size = self.wavelet_patchify(x)
        H, W = spatial_size
        B, N, C = tokens.shape
        if N != H * W:
            raise RuntimeError(f"Token count {N} does not match spatial dimensions {H} x {W}.")
        feat = tokens.transpose(1, 2).reshape(B, C, H, W)

        if self.use_sca and Mjoint.shape[-2:] != (H, W):
            Mjoint = F.interpolate(Mjoint, size=(H, W), mode="bilinear", align_corners=False)

        for i, block in enumerate(self.blocks):
            feat = block(feat)
            if self.use_sca and i in self.gate_layers:
                feat = self.gates[str(i)](feat, Mjoint)

        feat = self.pool(feat).flatten(1)
        return self.head(feat)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 1, 300, 300)

    print("\n========== WM-ConvMixer (H2+H3, corrected) Test ==========")
    for use_sca in [False, True]:
        model = WMConvMixer(
            in_channels=1, dim=128, depth=8, kernel_size=9,
            wavelet_patch_size=5, morphology_length=5,
            num_classes=4, use_sca=use_sca,
        )
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nuse_sca={use_sca}")
        print("Output shape:", tuple(out.shape))
        print("Param count: ", n_params)

    print("\nExpected output shape: [2, 4]")
    print("=================================================\n")