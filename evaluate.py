import argparse
import json
import os

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix

from config import get_device
from data import build_loaders
from models import MLPMixer
from utils import ensure_dir


def evaluate_split(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
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

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_split_report(result, classes, output_dir, split_name):
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    with open(
        os.path.join(output_dir, f"{split_name}_classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    np.savetxt(
        os.path.join(output_dir, f"{split_name}_confusion_matrix.csv"),
        confusion_matrix(result["y_true"], result["y_pred"]),
        fmt="%d",
        delimiter=",",
    )


def save_accuracy_comparison(train_accuracy, val_accuracy, output_path, checkpoint_epoch=None):
    """Save a simple SVG bar chart comparing clean train and validation accuracy."""
    width, height = 900, 560
    plot_left, plot_right = 150, 800
    plot_top, plot_bottom = 120, 450
    plot_height = plot_bottom - plot_top

    train_y = plot_bottom - (train_accuracy / 100.0) * plot_height
    val_y = plot_bottom - (val_accuracy / 100.0) * plot_height
    epoch_text = f" — checkpoint epoch {checkpoint_epoch}" if checkpoint_epoch is not None else ""

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="450" y="42" text-anchor="middle" font-family="Arial, sans-serif" font-size="26" font-weight="bold">Clean Training vs Validation Accuracy</text>
<text x="450" y="72" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#555">MLP-Mixer{epoch_text}</text>
<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333" stroke-width="2"/>
<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333" stroke-width="2"/>
'''
    for pct in (20, 40, 60, 80):
        y = plot_bottom - (pct / 100.0) * plot_height
        svg += f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#ddd"/>\\n'
    for pct in (0, 20, 40, 60, 80, 100):
        y = plot_bottom - (pct / 100.0) * plot_height + 5
        svg += f'<text x="130" y="{y:.1f}" text-anchor="end" font-family="Arial" font-size="13">{pct}%</text>\\n'

    train_x, val_x, bar_width = 275, 535, 170
    train_h = plot_bottom - train_y
    val_h = plot_bottom - val_y
    svg += f'<rect x="{train_x}" y="{train_y:.2f}" width="{bar_width}" height="{train_h:.2f}" fill="#4C78A8"/>\\n'
    svg += f'<rect x="{val_x}" y="{val_y:.2f}" width="{bar_width}" height="{val_h:.2f}" fill="#F58518"/>\\n'
    svg += f'<text x="360" y="{train_y - 14:.1f}" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{train_accuracy:.2f}%</text>\\n'
    svg += f'<text x="620" y="{val_y - 14:.1f}" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{val_accuracy:.2f}%</text>\\n'
    svg += '<text x="360" y="482" text-anchor="middle" font-family="Arial" font-size="18">Clean Train</text>\\n'
    svg += '<text x="620" y="482" text-anchor="middle" font-family="Arial" font-size="18">Clean Validation</text>\\n'
    gap = train_accuracy - val_accuracy
    svg += f'<text x="450" y="525" text-anchor="middle" font-family="Arial, sans-serif" font-size="16">Generalization gap: {gap:.2f} percentage points</text>\\n'
    svg += '</svg>\\n'

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing the CIFAR-100 dataset",
    )
    p.add_argument(
        "--output-dir",
        default="results/clean_train_val",
        help="Directory for clean train/validation evaluation results",
    )
    p.add_argument(
        "--clean-train-val",
        action="store_true",
        help="Evaluate the checkpoint on clean, unaugmented train and validation data",
    )
    args = p.parse_args()

    device = get_device(args.device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Please place the desired best.pt at that path."
        )

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = MLPMixer(**ckpt["model_config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Checkpoint val accuracy: {ckpt.get('val_acc', 'unknown')}")

    # augment=False is intentional here:
    # both the training split and validation split use the same clean
    # normalization-only transform, so their accuracies are directly comparable.
    train_loader, val_loader, test_loader, classes = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        augment=False,
    )

    criterion = nn.CrossEntropyLoss()

    if args.clean_train_val:
        ensure_dir(args.output_dir)

        train_result = evaluate_split(
            model, train_loader, criterion, device
        )
        val_result = evaluate_split(
            model, val_loader, criterion, device
        )

        comparison = {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val_accuracy": ckpt.get("val_acc"),
            "evaluation_type": "clean_train_vs_clean_validation",
            "augmentation": False,
            "train": {
                "loss": train_result["loss"],
                "accuracy": train_result["accuracy"],
                "correct": train_result["correct"],
                "total": train_result["total"],
            },
            "validation": {
                "loss": val_result["loss"],
                "accuracy": val_result["accuracy"],
                "correct": val_result["correct"],
                "total": val_result["total"],
            },
            "generalization_gap_percentage_points": (
                train_result["accuracy"] - val_result["accuracy"]
            ),
        }

        with open(
            os.path.join(args.output_dir, "clean_train_val_comparison.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        save_split_report(
            train_result, classes, args.output_dir, "clean_train"
        )
        save_split_report(
            val_result, classes, args.output_dir, "clean_val"
        )

        print("\nClean-data evaluation")
        print(
            f"Train loss: {train_result['loss']:.4f} | "
            f"Train accuracy: {train_result['accuracy']:.2f}%"
        )
        print(
            f"Val loss:   {val_result['loss']:.4f} | "
            f"Val accuracy:   {val_result['accuracy']:.2f}%"
        )
        print(
            f"Generalization gap: "
            f"{comparison['generalization_gap_percentage_points']:.2f} percentage points"
        )
        print(
            f"Saved comparison to: "
            f"{os.path.join(args.output_dir, 'clean_train_val_comparison.json')}"
        )
        return

    # Keep the original test-set evaluation behavior.
    result = evaluate_split(model, test_loader, criterion, device)

    print(f"Test loss: {result['loss']:.4f}")
    print(f"Test accuracy: {result['accuracy']:.2f}%")

    ensure_dir("results")
    np.savetxt(
        "results/confusion_matrix.csv",
        confusion_matrix(result["y_true"], result["y_pred"]),
        fmt="%d",
        delimiter=",",
    )
    report = classification_report(
        result["y_true"],
        result["y_pred"],
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    open(
        "results/classification_report.txt",
        "w",
        encoding="utf-8",
    ).write(report)
    print(report)


if __name__ == "__main__":
    main()
