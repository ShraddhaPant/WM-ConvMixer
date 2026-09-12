import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """
    Single-level 2D Haar Discrete Wavelet Transform.

    Input:
        [B, C, H, W]

    Output:
        [B, 4C, H/2, W/2]

    The four output subbands are:
        LL - Low-Low
        LH - Low-High
        HL - High-Low
        HH - High-High

    The DWT is the only spatial downsampling operation
    performed inside Wavelet Patchify.
    """

    def __init__(self):
        super().__init__()

        ll = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0]
            ]
        ) / 2.0

        lh = torch.tensor(
            [
                [-1.0, -1.0],
                [1.0, 1.0]
            ]
        ) / 2.0

        hl = torch.tensor(
            [
                [-1.0, 1.0],
                [-1.0, 1.0]
            ]
        ) / 2.0

        hh = torch.tensor(
            [
                [1.0, -1.0],
                [-1.0, 1.0]
            ]
        ) / 2.0

        filters = torch.stack(
            [ll, lh, hl, hh],
            dim=0
        ).unsqueeze(1)

        self.register_buffer(
            "filters",
            filters
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, 4C, H/2, W/2]
        """

        B, C, H, W = x.shape

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(
                "Haar DWT requires even spatial dimensions. "
                f"Received H={H}, W={W}."
            )

        filters = self.filters.repeat(
            C,
            1,
            1,
            1
        )

        x = F.conv2d(
            x,
            filters,
            stride=2,
            padding=0,
            groups=C
        )

        return x


class WaveletPatchify(nn.Module):
    """
    Wavelet Patchify for the WM-ConvMixer architecture.

    Expected H1 input:
        300 x 300

    Processing:

        300 x 300
             |
             | Haar DWT
             v
        150 x 150
             |
             | 5 x 5 wavelet patches
             v
          30 x 30
             |
             | flatten
             v
          900 tokens

    Therefore:

        Final spatial resolution = 30 x 30
        Number of tokens = 900

    Important:
        The DWT is the only spatial downsampling operation.
        The 5 x 5 operation groups the DWT coefficients into
        wavelet patches and produces the required 30 x 30 token grid.
    """

    def __init__(
        self,
        in_channels,
        embed_dim,
        patch_size=5
    ):
        super().__init__()

        if patch_size != 5:
            raise ValueError(
                "For the current H1/H2 configuration, "
                "patch_size must be 5."
            )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.dwt = HaarDWT2D()

        patch_dim = (
            4
            * in_channels
            * patch_size
            * patch_size
        )

        self.projection = nn.Linear(
            patch_dim,
            embed_dim
        )

    def forward(self, x):
        """
        Args:
            x:
                [B, C, 300, 300]

        Returns:
            tokens:
                [B, 900, embed_dim]

            spatial_size:
                (30, 30)
        """

        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape [B, C, H, W]. "
                f"Received shape {tuple(x.shape)}."
            )

        B, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {C}."
            )


        # STEP 1: Haar DWT
        #
        # 300 x 300 -> 150 x 150


        x = self.dwt(x)

        _, _, H_dwt, W_dwt = x.shape


        # STEP 2: Wavelet patchification
        #
        # 150 x 150 -> 30 x 30
        #
        # Each wavelet patch is 5 x 5.


        if H_dwt % self.patch_size != 0:
            raise ValueError(
                f"DWT height {H_dwt} is not divisible by "
                f"patch_size {self.patch_size}."
            )

        if W_dwt % self.patch_size != 0:
            raise ValueError(
                f"DWT width {W_dwt} is not divisible by "
                f"patch_size {self.patch_size}."
            )

        H_out = H_dwt // self.patch_size
        W_out = W_dwt // self.patch_size


        # Rearrange the wavelet coefficients into 5 x 5 patches.
        #
        # Before:
        # [B, 4C, 150, 150]
        #
        # After reshape:
        # [B, 4C, 30, 5, 30, 5]


        x = x.reshape(
            B,
            4 * C,
            H_out,
            self.patch_size,
            W_out,
            self.patch_size
        )


        # Rearrange dimensions:
        #
        # [B, 4C, 30, 5, 30, 5]
        #
        # ->
        #
        # [B, 30, 30, 4C, 5, 5]


        x = x.permute(
            0,
            2,
            4,
            1,
            3,
            5
        ).contiguous()


        # Flatten each wavelet patch.
        #
        # [B, 30, 30, 4C, 5, 5]
        #
        # ->
        #
        # [B, 30, 30, 4C*5*5]


        x = x.reshape(
            B,
            H_out,
            W_out,
            -1
        )


        # Linear projection.
        #
        # [B, 30, 30, patch_dim]
        #
        # ->
        #
        # [B, 30, 30, embed_dim]


        x = self.projection(x)


        # Flatten spatial grid into token sequence.
        #
        # [B, 30, 30, embed_dim]
        #
        # ->
        #
        # [B, 900, embed_dim]


        x = x.reshape(
            B,
            H_out * W_out,
            self.embed_dim
        )

        return x, (H_out, W_out)



# STEP 4 TEST


if __name__ == "__main__":

    B = 2
    C = 3
    EMBED_DIM = 128

    x = torch.randn(
        B,
        C,
        300,
        300
    )

    model = WaveletPatchify(
        in_channels=C,
        embed_dim=EMBED_DIM,
        patch_size=5
    )

    tokens, spatial_size = model(x)

    print()
    print("========== Wavelet Patchify Test ==========")
    print()
    print("Input shape:        ", x.shape)
    print("Token shape:        ", tokens.shape)
    print("Spatial resolution: ", spatial_size)
    print("Number of tokens:   ", tokens.shape[1])
    print()
    print("Expected:")
    print("Input:              [2, 3, 300, 300]")
    print("Tokens:             [2, 900, 128]")
    print("Spatial resolution: (30, 30)")
    print("============================================")
    print()