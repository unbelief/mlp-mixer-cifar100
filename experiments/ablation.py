"""Hardware-independent structural ablation.

Run from repository root:
    python experiments/ablation.py
"""
import copy
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import ModelConfig
from models import MLPMixer


def main():
    base = ModelConfig()
    experiments = []

    for patch in [2, 4, 8]:
        cfg = copy.deepcopy(base)
        cfg.patch_size = patch
        experiments.append((f"patch_{patch}", cfg))

    for depth in [2, 4, 6]:
        cfg = copy.deepcopy(base)
        cfg.depth = depth
        experiments.append((f"depth_{depth}", cfg))

    for dim in [64, 128, 192]:
        cfg = copy.deepcopy(base)
        cfg.embed_dim = dim
        cfg.token_dim = min(128, dim)
        cfg.channel_dim = dim * 2
        experiments.append((f"embed_{dim}", cfg))

    print(f"{'Experiment':<18}{'Tokens':>8}{'Embed':>8}{'Depth':>8}{'Params':>14}")
    for name, cfg in experiments:
        tokens = (cfg.image_size // cfg.patch_size) ** 2
        params = MLPMixer(**vars(cfg)).num_parameters()
        print(f"{name:<18}{tokens:>8}{cfg.embed_dim:>8}{cfg.depth:>8}{params:>14,}")


if __name__ == "__main__":
    main()
