"""Discriminator heads for FD representation-space GAN losses."""

import math

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ScalarFeatureHead(nn.Module):
    """Pooled-feature discriminator head.

    Shapes:
        feat:   [B, D]
        logits: [B]
    """

    def __init__(self, c_in: int, c_mid: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(c_in, c_mid)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(c_mid, c_mid)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(c_mid, 1)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 2:
            raise RuntimeError(f"ScalarFeatureHead expects [B, D], got {tuple(feat.shape)}")
        return self.net(feat.float()).squeeze(1)


class ViTPatchDiscriminatorHead(nn.Module):
    """Patch discriminator head for ViT token features.

    Shapes:
        tokens:       [B, N + P, C]
        patch_tokens: [B, N, C]
        logit_map:    [B, 1, H_p, W_p]
        logits:       [B]
    """

    def __init__(self, c_in: int, c_mid: int = 256, num_prefix_tokens: int = 1):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.norm = nn.LayerNorm(c_in)
        self.proj = spectral_norm(nn.Linear(c_in, c_mid))
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c_mid, 1, kernel_size=1)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"ViTPatchDiscriminatorHead expects [B, T, C], got {tuple(tokens.shape)}")
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]
        B, N, _ = patch_tokens.shape
        hw = int(math.isqrt(N))
        if hw * hw != N:
            raise RuntimeError(f"patch token count must be square, got N={N}")
        h = self.norm(patch_tokens)
        h = self.proj(h)
        h = h.transpose(1, 2).contiguous().reshape(B, -1, hw, hw)
        logit_map = self.conv(h)
        return logit_map.mean(dim=(1, 2, 3))


class ConvPatchDiscriminatorHead(nn.Module):
    """Patch discriminator head for CNN spatial feature maps.

    Shapes:
        fmap:      [B, C, H, W]
        logit_map: [B, 1, H, W]
        logits:    [B]
    """

    def __init__(self, c_in: int, c_mid: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(c_in, c_mid, kernel_size=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(c_mid, 1, kernel_size=1)),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        if fmap.ndim != 4:
            raise RuntimeError(f"ConvPatchDiscriminatorHead expects [B, C, H, W], got {tuple(fmap.shape)}")
        logit_map = self.net(fmap.float())
        return logit_map.mean(dim=(1, 2, 3))


def create_gan_head(feature_kind: str, c_in: int, c_mid: int, num_prefix_tokens: int = 0) -> nn.Module:
    if feature_kind == "pooled":
        return ScalarFeatureHead(c_in=c_in, c_mid=c_mid)
    if feature_kind == "vit_tokens":
        return ViTPatchDiscriminatorHead(
            c_in=c_in,
            c_mid=c_mid,
            num_prefix_tokens=num_prefix_tokens,
        )
    if feature_kind == "cnn_map":
        return ConvPatchDiscriminatorHead(c_in=c_in, c_mid=c_mid)
    raise ValueError(f"unknown GAN feature kind: {feature_kind}")
