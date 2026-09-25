import argparse
import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from config import get_device
from data import build_loaders
from models import MLPMixer
from utils import ensure_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MLPMixer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, loader, classes = build_loaders(
        batch_size=args.batch_size, num_workers=0, augment=False
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = correct = total = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            total_loss += criterion(logits, targets).item() * images.size(0)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += images.size(0)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    print(f"Test loss: {total_loss / total:.4f}")
    print(f"Test accuracy: {100 * correct / total:.2f}%")

    ensure_dir("results")
    np.savetxt("results/confusion_matrix.csv", confusion_matrix(y_true, y_pred),
               fmt="%d", delimiter=",")
    report = classification_report(y_true, y_pred, target_names=classes,
                                   digits=4, zero_division=0)
    open("results/classification_report.txt", "w", encoding="utf-8").write(report)
    print(report)


if __name__ == "__main__":
    main()
