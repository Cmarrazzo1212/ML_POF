"""
ML_POF_V25_H5_RUNNER.py

Purpose:
Train the TRNN classifier on your Pluto single-modulation OFDM HDF5 dataset.

Expected dataset folder:
C:>Users>usuario>Desktop>POF_researchPluto_Single_Mod_OFDM_Dataset_H5

Expected class folders:
BPSK_OFDM
QPSK_OFDM
8PSK_OFDM

Each .h5 file should contain:
- dataset named "X" with complex I/Q samples
- optional attribute "label"
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

class Config:
    # CHANGE THESE ONLY IF YOUR FOLDERS MOVE
    DATASET_DIR: Path = Path(r"C:\Users\usuario\Desktop\POF_research\Pluto_Single_Mod_OFDM_SYNC_Dataset_H5")
    MODEL_DIR: Path = Path(r"C:\Users\usuario\Desktop\POF_research\ML\Models")
    PLOT_DIR: Path = Path(r"C:\Users\usuario\Desktop\POF_research\ML\Plots")

    # New V25 model name. This does NOT overwrite your V24 model.
    CHECKPOINT: Path = MODEL_DIR / "memory_pluto_sync_v25.pt"

    CLASSES: List[str] = ["BPSK_OFDM", "QPSK_OFDM", "8PSK_OFDM"]
    NUM_CLASSES: int = 3

    # One training example = 512 complex samples = 2 x 512 real I/Q values
    SEQ_LEN: int =4096

    # 1024 gives about 64 windows from a 65536-sample file.
    # This matches your earlier 7680 windows/class expectation when there are 120 files/class.
    WINDOW_STEP: int = 1024

    BASE_FILTERS: int = 32
    DROPOUT: float = 0.25

    EPOCHS: int = 10
    BATCH_SIZE: int = 64
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.05

    EARLY_STOP_PATIENCE: int = 5
    SEED: int = 42

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# HDF5 DATA LOADING
# ============================================================

def normalize_complex_window(x: np.ndarray) -> np.ndarray:
    """
    Normalize each window by RMS power.
    This makes gain differences less dominant.
    """
    x = x.astype(np.complex64).reshape(-1)
    power = np.sqrt(np.mean(np.abs(x) ** 2))
    if power < 1e-12:
        power = 1e-12
    return x / power


def read_complex_x_from_h5(h5_path: Path) -> np.ndarray:
    """
    Reads the received Pluto samples from dataset X.
    """
    with h5py.File(h5_path, "r") as f:
        if "X" not in f:
            raise KeyError(f"{h5_path} does not contain dataset named X")
        x = f["X"][:]
    return np.asarray(x).reshape(-1).astype(np.complex64)


def file_label_from_path(h5_path: Path) -> str:
    """
    Uses the parent folder as the class label.
    Example:
    ...\BPSK_OFDM\BPSK_OFDM_txgain_-40_rxgain_30_capture_000.h5
    becomes:
    BPSK_OFDM
    """
    return h5_path.parent.name


def find_h5_files():
    files_by_class = defaultdict(list)

    for class_name in cfg.CLASSES:
        class_dir = cfg.DATASET_DIR / class_name
        h5_files = sorted(class_dir.glob("*.h5"))
        files_by_class[class_name].extend(h5_files)

    return files_by_class


def split_files_by_class(files_by_class):
    """
    File-level split.
    Windows from the same .h5 file stay together.
    This prevents validation leakage.
    """
    train_files = []
    val_files = []
    test_files = []

    for class_name, files in files_by_class.items():
        files = list(files)
        random.shuffle(files)

        n = len(files)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train_files.extend(files[:train_end])
        val_files.extend(files[train_end:val_end])
        test_files.extend(files[val_end:])

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)

    return train_files, val_files, test_files


def build_windows_from_files(file_list):
    X_list = []
    y_list = []
    file_name_list = []

    label_to_index = {name: i for i, name in enumerate(cfg.CLASSES)}

    for h5_path in file_list:
        class_name = file_label_from_path(h5_path)
        label = label_to_index[class_name]

        samples = read_complex_x_from_h5(h5_path)

        for start in range(0, len(samples) - cfg.SEQ_LEN + 1, cfg.WINDOW_STEP):
            window = samples[start:start + cfg.SEQ_LEN]
            window = normalize_complex_window(window)

            iq = np.stack(
                [window.real, window.imag],
                axis=0,
            ).astype(np.float32)

            X_list.append(iq)
            y_list.append(label)
            file_name_list.append(h5_path.name)

    X = torch.tensor(np.asarray(X_list), dtype=torch.float32)
    y = torch.tensor(np.asarray(y_list), dtype=torch.long)

    return X, y, file_name_list


def make_loader(X, y, shuffle):
    return DataLoader(
        TensorDataset(X, y),
        batch_size=cfg.BATCH_SIZE,
        shuffle=shuffle,
    )


def load_pluto_dataset():
    print("\n[data] Dataset folder:")
    print(cfg.DATASET_DIR)

    files_by_class = find_h5_files()

    print("\n[data] HDF5 files found:")
    total_files = 0
    for class_name in cfg.CLASSES:
        count = len(files_by_class[class_name])
        total_files += count
        print(f"  {class_name}: {count}")

    print(f"  TOTAL: {total_files}")

    if total_files == 0:
        raise FileNotFoundError("No .h5 files found. Check DATASET_DIR and class folders.")

    train_files, val_files, test_files = split_files_by_class(files_by_class)

    print("\n[data] File-level split:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Test files:  {len(test_files)}")

    X_train, y_train, _ = build_windows_from_files(train_files)
    X_val, y_val, _ = build_windows_from_files(val_files)
    X_test, y_test, test_file_names = build_windows_from_files(test_files)

    print("\n[data] Window counts:")
    print(f"  Train windows: {len(y_train)}")
    print(f"  Val windows:   {len(y_val)}")
    print(f"  Test windows:  {len(y_test)}")

    print("\n[data] Input tensor shape:")
    print(f"  X_train: {tuple(X_train.shape)}")
    print("  Expected: (windows, 2, 512)")

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)
    test_loader = make_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader, y_test, test_file_names


# ============================================================
# MODEL
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int = 3, drop: float = 0.0):
        super().__init__()
        padding = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=padding, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(ch, ch, k, padding=padding, bias=False),
            nn.BatchNorm1d(ch),
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class TripleSkipBlock(nn.Module):
    def __init__(self, ch: int, drop: float = 0.0):
        super().__init__()
        self.b1 = ResBlock(ch, drop=drop)
        self.b2 = ResBlock(ch, drop=drop)
        self.b3 = ResBlock(ch, drop=drop)

    def forward(self, x):
        return F.relu(self.b3(self.b2(self.b1(x))) + x)


class TRNN(nn.Module):
    def __init__(self):
        super().__init__()
        F_base = cfg.BASE_FILTERS
        D = cfg.DROPOUT

        self.stem = nn.Sequential(
            nn.Conv1d(2, F_base, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(F_base),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            TripleSkipBlock(F_base, D),
            nn.Conv1d(F_base, F_base * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(F_base * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.stage2 = nn.Sequential(
            TripleSkipBlock(F_base * 2, D),
            nn.Conv1d(F_base * 2, F_base * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(F_base * 4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.stage3 = nn.Sequential(
            TripleSkipBlock(F_base * 4, D),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F_base * 4, F_base * 2),
            nn.ReLU(),
            nn.Dropout(D),
            nn.Linear(F_base * 2, cfg.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


# ============================================================
# TRAINING
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for xb, yb in loader:
            xb = xb.to(cfg.DEVICE)
            yb = yb.to(cfg.DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * yb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
    }

    print("\n[train] Training started")
    print(f"[train] Device: {cfg.DEVICE}")
    print(f"[train] Model will save to: {cfg.CHECKPOINT}")
    print(
        f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>12} "
        f"{'Val Loss':>12} {'Val Acc':>12}"
    )
    print("-" * 65)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"{epoch:6d} {train_loss:12.4f} {train_acc * 100:11.2f}% "
            f"{val_loss:12.4f} {val_acc * 100:11.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), cfg.CHECKPOINT)
        else:
            patience_counter += 1

        if patience_counter >= cfg.EARLY_STOP_PATIENCE:
            print("\n[train] Early stopping triggered.")
            break

    print(f"\n[train] Best validation loss: {best_val_loss:.4f}")
    print(f"[train] Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"[train] Best model saved to: {cfg.CHECKPOINT}")

    return history


# ============================================================
# EVALUATION AND PLOTS
# ============================================================

def get_predictions(model, loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(cfg.DEVICE)
            logits = model(xb)
            pred = logits.argmax(1).cpu()
            preds.extend(pred.tolist())
            labels.extend(yb.tolist())

    return preds, labels


def plot_training(history):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy")
    plt.plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "v25_accuracy_curve.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "v25_loss_curve.png")
    plt.show()


def plot_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)

    print("\nConfusion Matrix:")
    print(cm)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("V25 Pluto HDF5 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1, 2], cfg.CLASSES, rotation=20)
    plt.yticks([0, 1, 2], cfg.CLASSES)

    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "v25_confusion_matrix.png")
    plt.show()


def main():
    print("=" * 70)
    print("ML_POF_V25_H5_RUNNER: Pluto Single-Mod OFDM HDF5 Training")
    print("=" * 70)
    print(f"Device: {cfg.DEVICE}")

    train_loader, val_loader, test_loader, y_test, test_file_names = load_pluto_dataset()

    model = TRNN().to(cfg.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] Total parameters: {total_params:,}")
    print(f"[model] Input shape: batch, 2, {cfg.SEQ_LEN}")

    history = train(model, train_loader, val_loader)

    print("\n[eval] Loading best model...")
    model.load_state_dict(torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE))

    preds, labels = get_predictions(model, test_loader)

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=cfg.CLASSES))

    #plot_training(history)
    #plot_confusion_matrix(preds, labels)

    print("\nDone.")
    print(f"Model saved to: {cfg.CHECKPOINT}")
    print(f"Plots saved to: {cfg.PLOT_DIR}")


if __name__ == "__main__":
    main()
