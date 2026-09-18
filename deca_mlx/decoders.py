"""Detail UV displacement generator matching official DECA ``Generator``."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .layers import apply_seq


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 181, out_channels: int = 1, out_scale: float = 0.01):
        super().__init__()
        self.out_scale = out_scale
        self.init_size = 8
        self.l1 = [nn.Linear(latent_dim, 128 * self.init_size ** 2)]
        # Indices must match the official Sequential so converted weights line up.
        # Official uses BatchNorm2d(128, 0.8) after the first block, i.e. eps=0.8.
        self.conv_blocks = [
            nn.BatchNorm(128),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(128, eps=0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(64, eps=0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(64, eps=0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(32, eps=0.8),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(16, eps=0.8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        ]

    def __call__(self, noise: mx.array) -> mx.array:
        out = apply_seq(self.l1, noise)
        # Official Generator views the linear output as NCHW (B, 128, 8, 8).
        out = out.reshape((out.shape[0], 128, self.init_size, self.init_size))
        out = out.transpose(0, 2, 3, 1)
        img = apply_seq(self.conv_blocks, out)
        return img * self.out_scale
