"""
train.py
========

Training and evaluation script for PEFTGuard models on matrix-format data.

Usage example:
--------------
```bash
python train.py \
    --base_model llama2_7b \
    --data_dirs /path/to/dataset1 /path/to/dataset2 \
    --output_dir ./outputs \
    --target_model q v
```

Arguments
~~~~~~~~~
* **base_model**   – backbone architecture to instantiate (see `model/` for supported names).
* **data_dirs**    – one or more dataset root directories, each containing `train/` and `test/` sub‑folders with `*.pth` tensors.
* **output_dir**   – directory where checkpoints and metrics will be saved.
* **target_model** – list of component tags (e.g. `q`, `k`, `v`, `o`) that the PEFT layer should adapt.

The script
~~~~~~~~~~
1. Constructs a `MatrixDataset` to lazily load `.pth` tensors and binary labels.
2. Creates balanced training/validation splits to keep class ratios intact.
3. Trains the selected PEFTGuard model with cosine‑annealed learning rate.
4. Tracks loss/accuracy/AUC on the validation set and applies early stopping.
5. Restores the best checkpoint and evaluates it on the held‑out test set.
"""

import argparse
import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Local model definitions -----------------------------------------------------
from model.PEFTGuard_llama2_7b import PEFTGuard_llama2_7b
from model.PEFTGuard_llama2_13b import PEFTGuard_llama2_13b
from model.PEFTGuard_qwen1_5 import PEFTGuard_qwen1_5
from model.PEFTGuard_glm import PEFTGuard_Glm2_6b
from model.PEFTGuard_roberta import PEFTGuard_roberta
from model.PEFTGuard_llama3_8b import PEFTGuard_llama3_8b
from model.PEFTGuard_vit import PEFTGuard_VIT_base
from model.PEFTGuard_qwenvl import PEFTGuard_qwenvl
from model.PEFTGuard_t5 import PEFTGuard_T5

# -----------------------------------------------------------------------------

class MatrixDataset(Dataset):
    """Dataset that lazily loads tensor files and binary labels.

    Each *directory* in ``directories`` must contain two sub‑folders: ``train``
    and ``test``. All ``*.pth`` files inside those sub‑folders are treated as
    individual samples. The binary label is inferred from the first character
    of the final underscore‑delimited token in the filename, e.g.::

        sample_0001_1.pth  -> label 1
        sample_0002_0.pth  -> label 0
    """

    def __init__(self, directories: List[str], subset: str, transform=None):
        self.directories = directories
        self.subset = subset  # "train" or "test"
        self.transform = transform
        self.files: List[str] = []
        self.labels: List[int] = []

        for directory in directories:
            subset_path = os.path.join(directory, subset)
            directory_files = glob.glob(os.path.join(subset_path, "*.pth"))
            directory_labels = [int(os.path.basename(path).split("_")[-1][0])
                                for path in directory_files]
            self.files.extend(directory_files)
            self.labels.extend(directory_labels)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.files)

    def __getitem__(self, idx: int):  # type: ignore[override]
        matrix_path = self.files[idx]
        matrix = torch.load(matrix_path)
        label = self.labels[idx]
        if self.transform:
            matrix = self.transform(matrix)
        return matrix, label


def balanced_split(dataset: MatrixDataset, val_size: float = 0.1) -> Tuple[Subset, Subset]:
    """Return train/val splits with preserved class distribution."""

    labels = np.array(dataset.labels)
    label0_indices = np.where(labels == 0)[0]
    label1_indices = np.where(labels == 1)[0]

    n_label0 = int(val_size * len(label0_indices))
    n_label1 = int(val_size * len(label1_indices))

    # Random stratified sampling
    val_indices = np.concatenate([
        np.random.choice(label0_indices, n_label0, replace=False),
        np.random.choice(label1_indices, n_label1, replace=False),
    ])
    train_indices = np.setdiff1d(np.arange(len(dataset)), val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def build_model(base_model: str, device: str, target_number: int):
    mapping = {
        "llama2_7b": PEFTGuard_llama2_7b,
        "llama2_13b": PEFTGuard_llama2_13b,
        "qwen1.5": PEFTGuard_qwen1_5,
        "glm2_6b": PEFTGuard_Glm2_6b,
        "roberta_base": PEFTGuard_roberta,
        "llama3_8b": PEFTGuard_llama3_8b,
        "vit_b": PEFTGuard_VIT_base,
        "qwenvl": PEFTGuard_qwenvl,
        "t5-base": PEFTGuard_T5,
    }
    if base_model not in mapping:
        raise ValueError(f"Unsupported base model: {base_model}")
    return mapping[base_model](device, target_number)

def main(args):
    """Entry point for training/validation/testing."""

    device = "cpu"  # change to "cuda" if GPU is available and desired

    target_number = len(args.target_model)
    model = build_model(args.base_model, device, target_number)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    train_dataset = MatrixDataset(args.data_dirs, "train")
    test_dataset = MatrixDataset(args.data_dirs, "test")
    train_subset, val_subset = balanced_split(train_dataset, val_size=0.1)

    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    # ------------------------------------------------------------------
    # Optimizer & scheduler
    # ------------------------------------------------------------------
    num_epochs = 20
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=5e-6)

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    early_stop_count = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in tqdm(range(num_epochs), desc="Epoch"):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False):
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=2).float().squeeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)
        scheduler.step()

        # ------------------------ Validation ------------------------
        model.eval()
        running_val_loss = 0.0
        correct, total = 0, 0
        all_pred, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                if isinstance(inputs, dict):
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(device)
                labels = labels.to(device)
                labels_one_hot = F.one_hot(labels, num_classes=2).float().squeeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels_one_hot)
                running_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_pred.extend(outputs[:, 1].cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_auc = roc_auc_score(all_labels, all_pred)

        # ------------------------ Early stopping --------------------
        improved = val_loss < best_val_loss and val_accuracy >= best_val_accuracy
        if improved:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

        if epoch >= num_epochs // 2:
            early_stop_count = early_stop_count + 1 if not improved else 0
            if (epoch + 1) == num_epochs or early_stop_count >= 5:
                print("Early stopping triggered – evaluating best model…")
                break

        print(f"Epoch {epoch + 1:02d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_acc={val_accuracy:.2f}% | "
              f"val_auc={val_auc:.4f}")

    # ------------------------------------------------------------------
    # Test evaluation with the best checkpoint
    # ------------------------------------------------------------------
    best_model = build_model(args.base_model, device, target_number)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()

    running_test_loss = 0.0
    correct, total = 0, 0
    all_pred, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test"):
            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=2).float().squeeze(1)

            outputs = best_model(inputs)
            loss = criterion(outputs, labels_one_hot)
            running_test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_pred.extend(outputs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    test_auc = roc_auc_score(all_labels, all_pred)

    print("\nTest summary")
    print("============")
    print(f"loss      : {test_loss:.4f}")
    print(f"accuracy  : {test_accuracy:.2f}%")
    print(f"roc_auc   : {test_auc:.4f}")

    # Persist metrics for downstream analysis
    with open(os.path.join(args.output_dir, "test_performance.txt"), "w") as fp:
        fp.write(f"Test Loss: {test_loss}\n")
        fp.write(f"Test Accuracy: {test_accuracy}%\n")
        fp.write(f"Test AUC: {test_auc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PEFTGuard model on matrix data.")

    parser.add_argument("--base_model", type=str, required=True,
                        help="Backbone architecture (e.g., 'llama2_7b').")
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True,
                        help="Dataset root directories containing 'train' and 'test' sub‑folders.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store checkpoints and reports.")
    parser.add_argument("--target_model", type=str, nargs="+",
                        default=["q", "v"], choices=["q", "k", "v", "o"],
                        help="PEFT components to adapt (choose any of q/k/v/o).")

    parsed_args = parser.parse_args()
    main(parsed_args)
