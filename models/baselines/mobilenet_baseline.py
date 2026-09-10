import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class LightweightCNNBaseline(nn.Module):
    """
    MobileNetV2-based lightweight CNN baseline for H1 (review Section IX-D).
    weights=None: trained from scratch on ELPV, matching ConvMixer's
    from-scratch training — no ImageNet pretraining on either side.

    width_mult scales channel width down from the ImageNet default (1.0)
    to approximate ConvMixer's parameter budget (~233,860) for the
    capacity-matched H1 comparison. The default width_mult=1.0 config
    is reserved separately for the deployment-matched comparison.
    """

    def __init__(self, in_channels=1, num_classes=4, width_mult=1.0):
        super().__init__()
        self.config = dict(width_mult=width_mult, in_channels=in_channels)

        base = mobilenet_v2(weights=None, num_classes=num_classes, width_mult=width_mult)

        # Replace first conv: 3-channel RGB stem -> 1-channel grayscale stem.
        old_conv = base.features[0][0]
        base.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        self.model = base

    def forward(self, x):
        return self.model(x)