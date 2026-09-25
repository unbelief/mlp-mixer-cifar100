from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 100
    embed_dim: int = 128
    depth: int = 4
    token_dim: int = 128
    channel_dim: int = 256
    dropout: float = 0.1


MODEL_PRESETS = {
    "tiny": ModelConfig(embed_dim=96, depth=3, token_dim=96, channel_dim=192),
    "small": ModelConfig(embed_dim=128, depth=4, token_dim=128, channel_dim=256),
    "base": ModelConfig(embed_dim=192, depth=6, token_dim=128, channel_dim=384),
}


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)
