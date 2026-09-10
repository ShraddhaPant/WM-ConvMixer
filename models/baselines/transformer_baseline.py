import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    """
    Small Vision Transformer baseline for H1 (review Section IX-D).
    Built from scratch, matching how ConvMixer and the CNN baseline
    are trained — no pretrained weights, no timm dependency, so
    parameter count stays under direct control.

    Patch embedding here is STANDARD strided-conv patchify, not
    wavelet — wavelet is reserved for H2, applied only on the
    ConvMixer branch (Section IX-D: "same ConvMixer backbone").
    """

    def __init__(self, in_channels=1, img_size=300, patch_size=15,
                 embed_dim=96, depth=2, num_heads=4, mlp_ratio=2,
                 num_classes=4):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        num_patches = (img_size // patch_size) ** 2

        self.config = dict(patch_size=patch_size, embed_dim=embed_dim,
                            depth=depth, num_heads=num_heads,
                            mlp_ratio=mlp_ratio, in_channels=in_channels)

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                      kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)               # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, num_patches, embed_dim]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])                 # CLS token output
        return self.head(x)