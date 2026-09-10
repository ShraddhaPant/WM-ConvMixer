import torch
import torch.nn as nn


class ConvMixerBlock(nn.Module):
    """
    One ConvMixer block: depthwise spatial mixing, then pointwise
    channel mixing. Residual path wraps only the depthwise branch,
    matching the original ConvMixer design (Trockman & Kolter, 2022)
    and review Section IX-B stage 3.
    """

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = x + self.depthwise(x)   # explicit residual, depthwise branch only
        x = self.pointwise(x)
        return x


class ConvMixer(nn.Module):
    """
    Standard ConvMixer backbone with ORIGINAL strided-conv patch
    embedding (no wavelet). This is the H1 baseline config — used to
    benchmark ConvMixer itself against CNN/ShuffleNet/Transformer
    before any WM-ConvMixer components (wavelet, SCA, LGAT) are added.

    Config must be reported for every experiment (review Section IX-B):
        dim, depth, kernel_size, patch_size all fixed and logged here.
    """

    def __init__(self, in_channels=1, dim=128, depth=8,
                 kernel_size=9, patch_size=10, num_classes=4):
        super().__init__()
        self.config = dict(dim=dim, depth=depth, kernel_size=kernel_size,
                            patch_size=patch_size, in_channels=in_channels)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )
        self.blocks = nn.Sequential(
            *[ConvMixerBlock(dim, kernel_size) for _ in range(depth)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.head(x)