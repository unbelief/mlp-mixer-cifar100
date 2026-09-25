import argparse
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import MODEL_PRESETS, get_device
from data import build_loaders
from models import MLPMixer
from utils import seed_everything, ensure_dir, save_json, save_history_plot


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = total_correct = total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in tqdm(loader, leave=False):
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(1) == targets).sum().item()
            total += images.size(0)
    return total_loss / total, 100.0 * total_correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_PRESETS.keys(), default="small")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    cfg = MODEL_PRESETS[args.model]
    model = MLPMixer(**vars(cfg)).to(device)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Trainable parameters: {model.num_parameters():,}")

    train_loader, val_loader, _, _ = build_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, augment=True
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ensure_dir("checkpoints")
    ensure_dir("results")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = -1.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.2f}% | val loss {val_loss:.4f} acc {val_acc:.2f}%")

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": vars(cfg),
            "model_preset": args.model,
            "epoch": epoch,
            "val_acc": val_acc,
            "history": history,
        }
        torch.save(checkpoint, "checkpoints/last.pt")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, "checkpoints/best.pt")

    elapsed = time.time() - start
    summary = {
        "device": str(device),
        "model": args.model,
        "parameters": model.num_parameters(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_accuracy": best_acc,
        "training_time_seconds": elapsed,
        "training_time_minutes": elapsed / 60,
    }
    save_json(summary, "results/training_summary.json")
    save_history_plot(history, "results/training_curves.png")


if __name__ == "__main__":
    main()
