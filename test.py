import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    roc_auc_score
)

from dataset import prepare_datasets


class CNN_LSTM_IDS(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.cnn = nn.Sequential(*list(base.children())[:-1])

        cnn_out = base.fc.in_features

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(128 * 2, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x


def save_confusion_matrix(all_labels, preds, output_dir):
    cm = confusion_matrix(all_labels, preds)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Benign", "Attack"],
        yticklabels=["Benign", "Attack"]
    )
    plt.title("Confusion Matrix (%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix_percent.png"), dpi=300)
    plt.close()


def save_roc_curve(all_labels, all_probs, output_dir):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CICIDS parquet dataset")
    parser.add_argument("--artifact_dir", type=str, default="artifacts", help="Where best_model.pth is saved")
    parser.add_argument("--output_dir", type=str, default="results", help="Where evaluation outputs will be saved")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=2)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    _, test_ds, meta, _ = prepare_datasets(
        data_dir=args.data_dir,
        scaler_save_path=None
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = CNN_LSTM_IDS().to(device)

    ckpt_path = os.path.join(args.artifact_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    accuracy = accuracy_score(all_labels, preds)
    roc_auc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, preds, digits=4)

    print("\n========== MODEL PERFORMANCE ==========\n")
    print(report)
    print("Accuracy :", accuracy)
    print("ROC-AUC  :", roc_auc)

    save_confusion_matrix(all_labels, preds, args.output_dir)
    save_roc_curve(all_labels, all_probs, args.output_dir)

    metrics = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "test_size": int(len(all_labels)),
        "meta": meta
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    print(f"\nResults saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
