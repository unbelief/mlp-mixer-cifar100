import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    """MLP-Mixer block. Input shape: [B, tokens, channels]."""

    def __init__(self, num_tokens, channels, token_dim, channel_dim, dropout):
        super().__init__()
        self.norm_tokens = nn.LayerNorm(channels)
        self.token_mixing = MLPBlock(num_tokens, token_dim, dropout)
        self.norm_channels = nn.LayerNorm(channels)
        self.channel_mixing = MLPBlock(channels, channel_dim, dropout)

    def forward(self, x):
        y = self.norm_tokens(x).transpose(1, 2)
        y = self.token_mixing(y).transpose(1, 2)
        x = x + y
        return x + self.channel_mixing(self.norm_channels(x))


class MLPMixer(nn.Module):
    def __init__(
        self, image_size=32, patch_size=4, in_channels=3, num_classes=100,
        embed_dim=128, depth=4, token_dim=128, channel_dim=256, dropout=0.1
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.num_tokens = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.blocks = nn.Sequential(*[
            MixerBlock(self.num_tokens, embed_dim, token_dim, channel_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
