from config import MODEL_PRESETS
from models import MLPMixer
import torch


def main():
    for name, cfg in MODEL_PRESETS.items():
        model = MLPMixer(**vars(cfg))
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"{name}: output={tuple(y.shape)}, params={model.num_parameters():,}")
        assert y.shape == (2, 100)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
