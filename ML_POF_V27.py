"""
ML_POF_V27_OFDM_FFT_GAIN_REPORT.py

Purpose:
Train on synchronized Pluto OFDM FFT-subcarrier features and report accuracy by TX gain.

Why V27 exists:
V26 proved FFT/subcarrier features improve Pluto classification.
V27 keeps the same core method, but adds experiment-friendly configuration and gain diagnostics.

Pipeline:
    H5 /X received samples
    -> split into OFDM symbols
    -> remove cyclic prefix
    -> FFT
    -> keep used subcarriers
    -> train classifier
    -> report accuracy by class and TX gain

Expected H5:
    /X complex samples, either compound r/i or normal complex
    root attributes such as label, tx_gain_db, rx_gain_db
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG - CHANGE EXPERIMENT SETTINGS HERE
# ============================================================

class Config:
    EXPERIMENT_NAME: str = "pluto_sync_fft_gain_report_v27"

    PROJECT_ROOT: Path = Path(r"C:\Users\usuario\Desktop\POF_research")
    DATASET_DIR: Path = PROJECT_ROOT / "Pluto_Single_Mod_OFDM_SYNC_Dataset_H5"
    MODEL_DIR: Path = PROJECT_ROOT / "ML" / "Models"
    PLOT_DIR: Path = PROJECT_ROOT / "ML" / "Plots"

    CHECKPOINT: Path = MODEL_DIR / "memory_pluto_sync_fft_gain_v27.pt"
    SUMMARY_CSV: Path = PLOT_DIR / "v27_accuracy_by_tx_gain.csv"

    CLASSES: List[str] = ["BPSK_OFDM", "QPSK_OFDM", "8PSK_OFDM"]

    FFT_SIZE: int = 64
    CP_LEN: int = 16
    USED_SUBCARRIERS: int = 48

    # One training example = this many complete OFDM symbols.
    OFDM_SYMBOLS_PER_WINDOW: int = 128
    WINDOW_STEP_SYMBOLS: int = 32

    EPOCHS: int = 25
    BATCH_SIZE: int = 32
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LABEL_SMOOTHING: float = 0.03
    EARLY_STOP_PATIENCE: int = 8
    SEED: int = 42

    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ============================================================
# H5 LOADING
# ============================================================

def read_h5_attrs(h5_path: Path) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as f:
        for key, value in f.attrs.items():
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="ignore")
            if isinstance(value, np.generic):
                value = value.item()
            attrs[key] = value
    return attrs


def get_tx_gain(attrs: Dict[str, Any]) -> float:
    for key in ["tx_gain_db", "TxGain_dB", "txGain", "tx_gain"]:
        if key in attrs:
            return float(attrs[key])
    return float("nan")


def read_complex_h5_dataset(h5_path: Path, dataset_name: str = "X") -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        x = f[dataset_name][()]

    if x.dtype.fields is not None:
        field_names = list(x.dtype.fields.keys())
        if "r" in field_names and "i" in field_names:
            x = x["r"].astype(np.float32) + 1j * x["i"].astype(np.float32)
        elif "real" in field_names and "imag" in field_names:
            x = x["real"].astype(np.float32) + 1j * x["imag"].astype(np.float32)
        else:
            raise ValueError(f"Unknown compound H5 fields in {h5_path}: {field_names}")
    else:
        x = np.asarray(x)

    return np.squeeze(x).astype(np.complex64)


def normalize_complex_vector(x: np.ndarray) -> np.ndarray:
    power = np.sqrt(np.mean(np.abs(x) ** 2))
    if power < 1e-12:
        power = 1e-12
    return (x / power).astype(np.complex64)


# ============================================================
# OFDM FEATURE EXTRACTION
# ============================================================

def used_subcarrier_indices(fft_size: int, used: int) -> np.ndarray:
    half = used // 2
    center = fft_size // 2
    neg = np.arange(center - half, center)
    pos = np.arange(center + 1, center + 1 + half)
    return np.concatenate([neg, pos])


def rx_to_ofdm_subcarriers(rx: np.ndarray) -> np.ndarray:
    symbol_len = cfg.FFT_SIZE + cfg.CP_LEN
    num_symbols = len(rx) // symbol_len

    if num_symbols < cfg.OFDM_SYMBOLS_PER_WINDOW:
        return np.empty((0, cfg.USED_SUBCARRIERS), dtype=np.complex64)

    rx = rx[: num_symbols * symbol_len]
    frames = rx.reshape(num_symbols, symbol_len)
    no_cp = frames[:, cfg.CP_LEN : cfg.CP_LEN + cfg.FFT_SIZE]
    freq = np.fft.fftshift(np.fft.fft(no_cp, axis=1), axes=1)

    idx = used_subcarrier_indices(cfg.FFT_SIZE, cfg.USED_SUBCARRIERS)
    used = freq[:, idx].astype(np.complex64)
    return normalize_complex_vector(used)


def make_windows_from_subcarriers(subc: np.ndarray) -> np.ndarray:
    windows = []
    n_symbols = subc.shape[0]

    for start in range(0, n_symbols - cfg.OFDM_SYMBOLS_PER_WINDOW + 1, cfg.WINDOW_STEP_SYMBOLS):
        block = subc[start : start + cfg.OFDM_SYMBOLS_PER_WINDOW, :]
        flat = block.reshape(-1)
        flat = normalize_complex_vector(flat)
        iq = np.stack([flat.real, flat.imag], axis=0).astype(np.float32)
        windows.append(iq)

    if len(windows) == 0:
        length = cfg.OFDM_SYMBOLS_PER_WINDOW * cfg.USED_SUBCARRIERS
        return np.empty((0, 2, length), dtype=np.float32)

    return np.stack(windows, axis=0)


# ============================================================
# DATASET
# ============================================================

def find_h5_files() -> List[Tuple[Path, int]]:
    all_items = []

    print("\n[data] Dataset folder:")
    print(cfg.DATASET_DIR)
    print("\n[data] HDF5 files found:")

    for label, class_name in enumerate(cfg.CLASSES):
        class_dir = cfg.DATASET_DIR / class_name
        files = sorted(class_dir.glob("*.h5"))
        print(f"  {class_name}: {len(files)}")
        for fp in files:
            all_items.append((fp, label))

    print(f"  TOTAL: {len(all_items)}")

    if len(all_items) == 0:
        raise RuntimeError("No HDF5 files found. Check DATASET_DIR and class folders.")

    return all_items


def split_files_by_class(items: List[Tuple[Path, int]]):
    groups = defaultdict(list)
    for fp, label in items:
        groups[label].append(fp)

    train, val, test = [], [], []

    for label, files in groups.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(0.70 * n)
        n_val = int(0.15 * n)

        train.extend((fp, label) for fp in files[:n_train])
        val.extend((fp, label) for fp in files[n_train:n_train + n_val])
        test.extend((fp, label) for fp in files[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print("\n[data] File-level split:")
    print(f"  Train files: {len(train)}")
    print(f"  Val files:   {len(val)}")
    print(f"  Test files:  {len(test)}")

    return train, val, test


def build_tensor_dataset(file_items: List[Tuple[Path, int]], split_name: str):
    X_list = []
    y_list = []
    gain_list = []
    file_list = []

    for fp, label in file_items:
        attrs = read_h5_attrs(fp)
        tx_gain = get_tx_gain(attrs)

        rx = read_complex_h5_dataset(fp, "X")
        subc = rx_to_ofdm_subcarriers(rx)
        windows = make_windows_from_subcarriers(subc)

        if windows.shape[0] == 0:
            print(f"[warning] No windows made from {fp.name}")
            continue

        X_list.append(windows)
        y_list.extend([label] * windows.shape[0])
        gain_list.extend([tx_gain] * windows.shape[0])
        file_list.extend([fp.name] * windows.shape[0])

    if len(X_list) == 0:
        raise RuntimeError(f"No windows were created for split: {split_name}")

    X = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    gains = np.asarray(gain_list, dtype=np.float32)

    print(f"  {split_name} windows: {X.shape[0]}")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
        torch.tensor(gains, dtype=torch.float32),
        file_list,
    )


def build_loaders():
    items = find_h5_files()
    train_items, val_items, test_items = split_files_by_class(items)

    print("\n[data] Creating OFDM FFT-subcarrier windows:")
    X_train, y_train, gain_train, _ = build_tensor_dataset(train_items, "Train")
    X_val, y_val, gain_val, _ = build_tensor_dataset(val_items, "Val")
    X_test, y_test, gain_test, test_files = build_tensor_dataset(test_items, "Test")

    input_len = cfg.OFDM_SYMBOLS_PER_WINDOW * cfg.USED_SUBCARRIERS

    print("\n[data] Input tensor shape:")
    print(f"  X_train: {tuple(X_train.shape)}")
    print(f"  Expected: (windows, 2, {input_len})")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test, gain_test), batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, y_test, gain_test, test_files


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
    def __init__(self, ch: int, drop: float = 0.20):
        super().__init__()
        self.b1 = ResBlock(ch, drop=drop)
        self.b2 = ResBlock(ch, drop=drop)
        self.b3 = ResBlock(ch, drop=drop)

    def forward(self, x):
        return F.relu(self.b3(self.b2(self.b1(x))) + x)


class TRNN(nn.Module):
    def __init__(self):
        super().__init__()
        F_base = 32
        D = 0.20

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
            nn.Linear(F_base * 2, len(cfg.CLASSES)),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            xb = batch[0].to(cfg.DEVICE)
            yb = batch[1].to(cfg.DEVICE)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print("\n[train] Training started")
    print(f"[train] Device: {cfg.DEVICE}")
    print(f"[train] Model will save to: {cfg.CHECKPOINT}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>12} {'Val Loss':>12} {'Val Acc':>12}")
    print("-" * 65)

    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)

        print(f"{epoch:6d} {train_loss:12.4f} {train_acc * 100:11.2f}% {val_loss:12.4f} {val_acc * 100:11.2f}%")

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


def evaluate(model, test_loader):
    model.eval()
    preds, labels, gains = [], [], []

    with torch.no_grad():
        for xb, yb, gb in test_loader:
            xb = xb.to(cfg.DEVICE)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().tolist())
            labels.extend(yb.tolist())
            gains.extend(gb.tolist())

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=cfg.CLASSES))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    print_accuracy_by_tx_gain(labels, preds, gains)
    save_accuracy_by_tx_gain_csv(labels, preds, gains)


def print_accuracy_by_tx_gain(labels, preds, gains):
    print("\nAccuracy by TX gain:")
    print("-" * 60)
    print(f"{'TX Gain':>10} {'Total':>8} {'Accuracy':>10} {'BPSK':>10} {'QPSK':>10} {'8PSK':>10}")

    unique_gains = sorted(set(round(float(g), 3) for g in gains))
    for gain in unique_gains:
        idxs = [i for i, g in enumerate(gains) if round(float(g), 3) == gain]
        total = len(idxs)
        correct = sum(1 for i in idxs if preds[i] == labels[i])
        acc = correct / total if total else 0.0

        class_accs = []
        for class_id in range(len(cfg.CLASSES)):
            cidxs = [i for i in idxs if labels[i] == class_id]
            if len(cidxs) == 0:
                class_accs.append("   n/a")
            else:
                ccorrect = sum(1 for i in cidxs if preds[i] == labels[i])
                class_accs.append(f"{100 * ccorrect / len(cidxs):6.1f}%")

        print(f"{gain:10.1f} {total:8d} {100 * acc:9.1f}% {class_accs[0]:>10} {class_accs[1]:>10} {class_accs[2]:>10}")


def save_accuracy_by_tx_gain_csv(labels, preds, gains):
    unique_gains = sorted(set(round(float(g), 3) for g in gains))

    with open(cfg.SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_gain_db", "class", "correct", "total", "accuracy_percent"])

        for gain in unique_gains:
            idxs_gain = [i for i, g in enumerate(gains) if round(float(g), 3) == gain]

            correct_all = sum(1 for i in idxs_gain if preds[i] == labels[i])
            total_all = len(idxs_gain)
            acc_all = 100 * correct_all / total_all if total_all else 0.0
            writer.writerow([gain, "ALL", correct_all, total_all, acc_all])

            for class_id, class_name in enumerate(cfg.CLASSES):
                idxs = [i for i in idxs_gain if labels[i] == class_id]
                correct = sum(1 for i in idxs if preds[i] == labels[i])
                total = len(idxs)
                acc = 100 * correct / total if total else 0.0
                writer.writerow([gain, class_name, correct, total, acc])

    print(f"\n[report] Saved TX-gain accuracy CSV to: {cfg.SUMMARY_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("ML_POF_V27: OFDM FFT + TX Gain Report")
    print("=" * 70)
    print(f"Experiment: {cfg.EXPERIMENT_NAME}")
    print(f"Device: {cfg.DEVICE}")

    train_loader, val_loader, test_loader, y_test, gain_test, test_files = build_loaders()

    model = TRNN().to(cfg.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())

    input_len = cfg.OFDM_SYMBOLS_PER_WINDOW * cfg.USED_SUBCARRIERS
    print(f"\n[model] Total parameters: {total_params:,}")
    print(f"[model] Input shape: batch, 2, {input_len}")

    train(model, train_loader, val_loader)

    print("\n[eval] Loading best model...")
    model.load_state_dict(torch.load(cfg.CHECKPOINT, map_location=cfg.DEVICE))
    evaluate(model, test_loader)

    print("\nDone.")
    print(f"Model saved to: {cfg.CHECKPOINT}")


if __name__ == "__main__":
    main()
