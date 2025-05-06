"""
contrast_train.py
=================

Train **PEFTGuard‑LLaMA‑2‑7B** on matrix‑format tensors with an optional
supervised‑contrastive loss.

Only the *llama2_7b* backbone is supported in this minimal version; all other
model references have been removed for simplicity.
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Model definition (local file: model/PEFTGuard_llama2_7b.py)
# -----------------------------------------------------------------------------

import torch.nn as nn


class PEFTGuard_llama2_7b(nn.Module):
    """Toy CNN that maps 4096×4096×C matrices to binary logits."""

    def __init__(self, device: str, target_number: int = 2):
        super().__init__()
        in_channels = target_number * 32
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=8)

        flattened = 512 * 512 * 16
        self.fc1 = nn.Linear(flattened, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.to(device)

    def forward(self, x):  # type: ignore[override]
        x = x.view(-1, self.conv1.in_channels, 4096, 4096)
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        embeddings = F.leaky_relu(self.fc2(x))
        logits = self.fc3(embeddings)
        return logits, embeddings


# -----------------------------------------------------------------------------
# Loss: Supervised Contrastive (ICML 2020)
# -----------------------------------------------------------------------------

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        device = features.device
        bsz = features.size(0)
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        eye = torch.eye(bsz, dtype=torch.bool, device=device)
        labels = labels.view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float()
        sim.masked_fill_(eye, -9e15)  # exclude self

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)
        denom = label_mask.sum(1) - 1
        mean_log_prob = (label_mask * log_prob).sum(1) / (denom + 1e-8)
        return -mean_log_prob.mean()


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

class MatrixDataset(Dataset):
    """Streams ``.pth`` tensors and binary labels inferred from filenames."""

    def __init__(self, directories: List[str], subset: str):
        self.files: List[str] = []
        self.labels: List[int] = []
        for directory in directories:
            subset_root = os.path.join(directory, subset)
            paths = glob.glob(os.path.join(subset_root, "*.pth"))
            self.files.extend(paths)
            self.labels.extend([int(os.path.basename(p).split("_")[-1][0]) for p in paths])

    def __len__(self):  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx):  # type: ignore[override]
        x = torch.load(self.files[idx])
        y = self.labels[idx]
        return x, y


def balanced_split(ds: MatrixDataset, val_ratio: float = 0.1) -> Tuple[Subset, Subset]:
    labels = np.array(ds.labels)
    idx0, idx1 = np.where(labels == 0)[0], np.where(labels == 1)[0]
    n0, n1 = int(len(idx0) * val_ratio), int(len(idx1) * val_ratio)

    val_idx = np.concatenate([
        np.random.choice(idx0, n0, replace=False),
        np.random.choice(idx1, n1, replace=False),
    ])
    train_idx = np.setdiff1d(np.arange(len(ds)), val_idx)
    return Subset(ds, train_idx.tolist()), Subset(ds, val_idx.tolist())


# -----------------------------------------------------------------------------
# Training / evaluation loop
# -----------------------------------------------------------------------------

def main(args):  # noqa: C901
    device = "cpu"
    target_number = len(args.target_model)

    model = PEFTGuard_llama2_7b(device, target_number)
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds = MatrixDataset(args.data_dirs, "train")
    test_ds = MatrixDataset(args.data_dirs, "test")
    train_sub, val_sub = balanced_split(train_ds)

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    epochs = 20
    loss_cls = torch.nn.BCEWithLogitsLoss()
    loss_supcon = SupConLoss()
    alpha = 0.1

    optimiser = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=5e-6)

    best_val_loss, best_acc = float("inf"), 0.0
    ckpt = os.path.join(args.output_dir, "best_model.pth")

    for epoch in tqdm(range(epochs), desc="Epoch"):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, leave=False):
            x, y = x.to(device), y.to(device)
            y_oh = F.one_hot(y, num_classes=2).float().squeeze(1)

            optimiser.zero_grad()
            logits, embeds = model(x)
            loss = loss_cls(logits, y_oh) + alpha * loss_supcon(embeds, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        sched.step()

        # Validation -----------------------------------------------------
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                y_oh = F.one_hot(y, num_classes=2).float().squeeze(1)
                logits, _ = model(x)
                val_loss += loss_cls(logits, y_oh).item()
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
                preds.extend(logits[:, 1].cpu().numpy())
                labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        val_auc = roc_auc_score(labels, preds)

        if val_loss < best_val_loss and val_acc >= best_acc:
            best_val_loss, best_acc = val_loss, val_acc
            torch.save(model.state_dict(), ckpt)
        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | val_auc={val_auc:.4f}")

    # Testing --------------------------------------------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    test_loss, correct, total = 0.0, 0, 0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(device), y.to(device)
            y_oh = F.one_hot(y, num_classes=2).float().squeeze(1)
            logits, _ = model(x)
            test_loss += loss_cls(logits, y_oh).item()
            _, pred = torch.max(logits, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            preds.extend(logits[:, 1].cpu().numpy())
            labels.extend(y.cpu().numpy())
    test_loss /= len(test_loader)
    test_acc = 100 * correct / total
    test_auc = roc_auc_score(labels, preds)

    print("\nTest summary\n------------")
    print(f"loss     : {test_loss:.4f}")
    print(f"accuracy : {test_acc:.2f}%")
    print(f"roc_auc  : {test_auc:.4f}")

    with open(os.path.join(args.output_dir, "test_performance.txt"), "w") as fp:
        fp.write(f"Test Loss: {test_loss}\n")
        fp.write(f"Test Accuracy: {test_acc}%\n")
        fp.write(f"Test AUC: {test_auc}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Contrastive training for PEFTGuard‑LLaMA‑2‑7B.")
    p.add_argument("--data_dirs", nargs="+", required=True,
                   help="Dataset roots containing 'train' and 'test'.")
    p.add_argument("--output_dir", required=True, help="Directory for outputs.")
    p.add_argument("--target_model", nargs="+", default=["q", "v"], choices=["q", "k", "v", "o"],
                   help="PEFT component tags; affects input channel count.")
    main(p.parse_args())
