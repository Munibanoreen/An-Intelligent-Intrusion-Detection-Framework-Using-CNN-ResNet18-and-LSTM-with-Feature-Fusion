import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, roc_auc_score

from dataset import prepare_datasets


class CNN_LSTM_IDS(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(weights=None)

        # same as your code: first conv changed to 1-channel
        base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # remove classifier, keep CNN feature extractor
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        cnn_out = base.fc.in_features  # 512 for resnet18

        # same architecture: LSTM after CNN feature vector
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # same binary output
        self.fc = nn.Linear(128 * 2, 1)

    def forward(self, x):
        x = self.cnn(x)               # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        x = x.unsqueeze(1)            # (B, 1, 512)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]        # (B, 256)
        x = self.fc(x)                # (B, 1)
        return x


def evaluate(model, loader, device):
    model.eval()

    all_probs = []
    all_labels = []
    running_loss = 0.0

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)

            running_loss += loss.item()
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    preds = (all_probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": running_loss / len(loader),
        "auc": auc,
        "preds": preds,
        "probs": all_probs,
        "labels": all_labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CICIDS parquet dataset")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Directory to save model/checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds, test_ds, meta, scaler = prepare_datasets(
        data_dir=args.data_dir,
        scaler_save_path=os.path.join(args.output_dir, "scaler.pkl")
    )

    with open(os.path.join(args.output_dir, "data_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model = CNN_LSTM_IDS().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_auc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(train_loss=running_loss / max(1, pbar.n + 1))

        train_loss = running_loss / len(train_loader)

        eval_out = evaluate(model, test_loader, device)
        val_loss = eval_out["loss"]
        val_auc = eval_out["auc"]
        preds = eval_out["preds"]
        labels = eval_out["labels"]

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val AUC   : {val_auc:.4f}")
        print(classification_report(labels.astype(int), preds, digits=4))

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_auc": float(val_auc)
        })

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_auc": best_auc,
                    "meta": meta
                },
                ckpt_path
            )
            print(f"Best model saved to: {ckpt_path}")

    with open(os.path.join(args.output_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining complete.")
    print(f"Best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
