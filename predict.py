import argparse
import torch
from torchvision import datasets, transforms

from config import get_device
from data import CIFAR100_MEAN, CIFAR100_STD
from models import MLPMixer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MLPMixer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    dataset = datasets.CIFAR100("data", train=False, download=True, transform=transform)
    image, target = dataset[args.index]

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    pred = logits.argmax(1).item()

    print(f"True label: {dataset.classes[target]}")
    print(f"Predicted:  {dataset.classes[pred]}")
    print(f"Confidence: {logits.softmax(1)[0, pred].item():.4f}")


if __name__ == "__main__":
    main()
