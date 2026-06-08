"""
ML_POF_V2.py

Research-friendly CNN pipeline for OFDM / RF signal classification using H5 IQ data.

Main behavior:
1. Scans an extracted dataset folder containing class folders such as BPSK_8PSK.
2. Reads .h5 files containing complex IQ signals.
3. Shows grouped RF analysis plots across detected classes.
4. Splits long IQ recordings into fixed-size windows.
5. Trains a 1D CNN classifier with PyTorch.
6. Resumes from the shared checkpoint if it exists:
       ML/Models/iq_cnn_classifier.pt
7. Saves ONLY the best CNN checkpoint to that same .pt file.
8. Does NOT save PNG graph files; plots are displayed only.

Spyder quick test:
--data "C:/Users/usuario/Desktop/POF_research/OFDM Modulation Classification Dataset" --epochs 1 --max-windows-per-dataset 10 --window-size 1024

Longer run:
--data "C:/Users/usuario/Desktop/POF_research/OFDM Modulation Classification Dataset" --epochs 10 --max-windows-per-dataset 1000 --window-size 2048
"""

import argparse
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

MODEL_PATH = "ML/Models/iq_cnn_classifier.pt"


# ============================================================
# Basic helpers
# ============================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def extract_snr_from_filename(filename: str) -> Optional[int]:
    match = re.search(r"(-?\d+)\s*dB", filename)
    if match:
        return int(match.group(1))
    return None


def find_h5_files(data_root: str) -> List[Tuple[str, str, str, Optional[int]]]:
    records = []
    for root, _, files in os.walk(data_root):
        h5_files = [f for f in files if f.lower().endswith(".h5")]
        if not h5_files:
            continue

        class_label = os.path.basename(root)
        for file_name in sorted(h5_files):
            file_path = os.path.join(root, file_name)
            snr = extract_snr_from_filename(file_name)
            records.append((file_path, class_label, file_name, snr))
    return records


def read_first_dataset_from_h5(file_path: str) -> np.ndarray:
    with h5py.File(file_path, "r") as f:
        key = list(f.keys())[0]
        data = f[key][:]
    return np.asarray(data).reshape(-1)


def normalize_iq(iq: np.ndarray) -> np.ndarray:
    iq = iq.astype(np.complex64)
    power = np.mean(np.abs(iq) ** 2)
    if power > 0:
        iq = iq / np.sqrt(power)
    return iq


def iq_to_two_channels(iq_window: np.ndarray) -> np.ndarray:
    return np.stack([np.real(iq_window), np.imag(iq_window)], axis=0).astype(np.float32)


def choose_representative_records(records: List[Tuple[str, str, str, Optional[int]]], max_classes: int) -> List[Tuple[str, str, str, Optional[int]]]:
    by_class = defaultdict(list)
    for rec in records:
        by_class[rec[1]].append(rec)

    preferred_snrs = [10, 12, 14, 16, 18, 20, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
    chosen_records = []

    for label in sorted(by_class.keys()):
        class_records = by_class[label]
        chosen = None
        for snr in preferred_snrs:
            matches = [r for r in class_records if r[3] == snr]
            if matches:
                chosen = matches[0]
                break
        if chosen is None:
            chosen = class_records[0]
        chosen_records.append(chosen)
        if len(chosen_records) >= max_classes:
            break

    return chosen_records


def load_preview_signals(records: List[Tuple[str, str, str, Optional[int]]], max_classes: int, max_samples: int) -> Dict[str, np.ndarray]:
    preview = {}
    chosen_records = choose_representative_records(records, max_classes=max_classes)
    for file_path, label, file_name, snr in chosen_records:
        iq = read_first_dataset_from_h5(file_path)
        iq = normalize_iq(iq[:max_samples])
        label_text = label
        if snr is not None:
            label_text += f" ({snr} dB)"
        preview[label_text] = iq
        print(f"  Preview signal: {label_text} from {file_name}")
    return preview


# ============================================================
# RF signal-analysis plots, grouped and overlaid
# ============================================================

def plot_rf_signal_overview(preview: Dict[str, np.ndarray], time_samples: int = 800, freq_samples: int = 8192) -> None:
    """One grouped figure: time-domain, power, FFT, and PSD overlays."""
    if not preview:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_time, ax_power, ax_fft, ax_psd = axes.ravel()

    for label, iq in preview.items():
        short = iq[: min(len(iq), time_samples)]
        seg = iq[: min(len(iq), freq_samples)]

        # Real component only for readability. Constellation plot still shows I/Q.
        ax_time.plot(np.real(short), linewidth=1.0, label=label)

        power = np.abs(short) ** 2
        ax_power.plot(power, linewidth=1.0, label=label)

        window = np.hanning(len(seg))
        spectrum = np.fft.fftshift(np.fft.fft(seg * window))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(seg), d=1.0))
        mag_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        mag_db = mag_db - np.max(mag_db)
        ax_fft.plot(freqs, mag_db, linewidth=1.0, label=label)

        psd = (np.abs(spectrum) ** 2) / max(len(seg), 1)
        psd_db = 10 * np.log10(psd + 1e-12)
        psd_db = psd_db - np.max(psd_db)
        ax_psd.plot(freqs, psd_db, linewidth=1.0, label=label)

    ax_time.set_title("Time domain overlay: real/I component")
    ax_time.set_xlabel("Sample")
    ax_time.set_ylabel("Normalized amplitude")
    ax_time.grid(True)

    ax_power.set_title("Instantaneous power overlay")
    ax_power.set_xlabel("Sample")
    ax_power.set_ylabel("Power |IQ|²")
    ax_power.grid(True)

    ax_fft.set_title("FFT magnitude overlay, normalized")
    ax_fft.set_xlabel("Normalized frequency")
    ax_fft.set_ylabel("Magnitude relative to peak (dB)")
    ax_fft.set_ylim(-80, 5)
    ax_fft.grid(True)

    ax_psd.set_title("Power spectral density overlay, normalized")
    ax_psd.set_xlabel("Normalized frequency")
    ax_psd.set_ylabel("PSD relative to peak (dB)")
    ax_psd.set_ylim(-80, 5)
    ax_psd.grid(True)

    handles, labels = ax_time.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("RF Signal Overview Across Detected Classes", fontsize=14)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()


def plot_constellation_grid(preview: Dict[str, np.ndarray], max_points: int = 3000) -> None:
    """Constellations in one grid instead of many separate windows."""
    if not preview:
        return

    n = len(preview)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows))
    axes = np.asarray(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, (label, iq) in zip(axes, preview.items()):
        step = max(1, len(iq) // max_points)
        pts = iq[::step]
        ax.scatter(np.real(pts), np.imag(pts), s=3, alpha=0.35)
        ax.set_title(label)
        ax.set_xlabel("I / real")
        ax.set_ylabel("Q / imaginary")
        ax.grid(True)
        ax.axis("equal")
        ax.axis("on")

    fig.suptitle("Constellation / IQ Scatter by Class", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_autocorrelation_and_distributions(preview: Dict[str, np.ndarray], samples: int = 8192, max_lag: int = 300) -> None:
    """One grouped figure for autocorrelation, amplitude distribution, and phase distribution."""
    if not preview:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_ac, ax_amp, ax_phase = axes

    for label, iq in preview.items():
        seg = iq[: min(len(iq), samples)]

        autocorr = np.correlate(seg, seg, mode="full")
        autocorr = autocorr[len(autocorr) // 2:]
        autocorr = np.abs(autocorr)
        if np.max(autocorr) > 0:
            autocorr = autocorr / np.max(autocorr)
        ax_ac.plot(autocorr[:max_lag], linewidth=1.0, label=label)

        amp = np.abs(seg)
        ax_amp.hist(amp, bins=60, density=True, histtype="step", linewidth=1.2, label=label)

        phase = np.angle(seg)
        ax_phase.hist(phase, bins=60, density=True, histtype="step", linewidth=1.2, label=label)

    ax_ac.set_title("Autocorrelation magnitude overlay")
    ax_ac.set_xlabel("Lag")
    ax_ac.set_ylabel("Normalized correlation")
    ax_ac.grid(True)

    ax_amp.set_title("Amplitude distribution overlay")
    ax_amp.set_xlabel("Amplitude")
    ax_amp.set_ylabel("Density")
    ax_amp.grid(True)

    ax_phase.set_title("Phase distribution overlay")
    ax_phase.set_xlabel("Phase (radians)")
    ax_phase.set_ylabel("Density")
    ax_phase.grid(True)

    handles, labels = ax_ac.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("Structure and Distribution Analysis Across Classes", fontsize=14)
    plt.tight_layout(rect=[0, 0.12, 1, 0.92])
    plt.show()


def plot_spectrogram_example(preview: Dict[str, np.ndarray], samples: int = 32768) -> None:
    """Spectrograms get dense, so show only one representative example."""
    if not preview:
        return

    label = list(preview.keys())[0]
    iq = preview[label]
    seg = iq[: min(len(iq), samples)]

    plt.figure(figsize=(11, 5))
    plt.specgram(np.real(seg), NFFT=512, Fs=1.0, noverlap=384)
    plt.title(f"Spectrogram, zoomed representative example: {label}")
    plt.xlabel("Time window")
    plt.ylabel("Normalized frequency")
    plt.colorbar(label="Power")
    plt.tight_layout()
    plt.show()


def plot_rf_analysis(records: List[Tuple[str, str, str, Optional[int]]], max_classes: int, preview_samples: int) -> None:
    print_header("Step 1B: Grouped RF Signal Analysis Plots")
    print("Plots are displayed only. PNG files are not saved.")
    print("Using one representative H5 file per detected class.")

    preview = load_preview_signals(records, max_classes=max_classes, max_samples=preview_samples)
    plot_rf_signal_overview(preview)
    plot_constellation_grid(preview)
    plot_autocorrelation_and_distributions(preview)
    plot_spectrogram_example(preview)


# ============================================================
# Dataset creation
# ============================================================

class IQWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_window_dataset(
    records: List[Tuple[str, str, str, Optional[int]]],
    label_to_idx: Dict[str, int],
    window_size: int,
    stride: int,
    max_windows_per_dataset: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []
    class_counts = Counter()

    print_header("Step 2: Building Training Windows")
    print("Converting long complex IQ signals into smaller training windows...")

    total_files = len(records)
    for idx, (file_path, label, file_name, snr) in enumerate(records, start=1):
        iq = read_first_dataset_from_h5(file_path)
        iq = normalize_iq(iq)

        n_possible = 1 + max(0, (len(iq) - window_size) // stride)
        n_to_use = n_possible
        if max_windows_per_dataset is not None:
            n_to_use = min(n_to_use, max_windows_per_dataset)

        if n_to_use <= 0:
            continue

        if n_possible <= n_to_use:
            starts = np.arange(n_possible) * stride
        else:
            starts = np.linspace(0, len(iq) - window_size, n_to_use).astype(int)

        for start in starts:
            window = iq[start:start + window_size]
            X_list.append(iq_to_two_channels(window))
            y_list.append(label_to_idx[label])
            class_counts[label] += 1

        if idx == 1 or idx == total_files or idx % 10 == 0:
            print(f"Progress: {idx}/{total_files} H5 files processed")

    if not X_list:
        raise RuntimeError("No training windows were created. Try a smaller --window-size.")

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    print("\nClass/window counts:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts[label]}")

    print(f"\nTotal windows: {len(y)}")
    print(f"Window shape: {X.shape[1:]} meaning [I/Q channels, samples]")
    return X, y


# ============================================================
# CNN model - kept compatible with the original visual CNN
# ============================================================

class IQCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============================================================
# Training / evaluation
# ============================================================

def run_one_epoch(model, loader, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            if train:
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * len(y_batch)
        total_correct += (preds == y_batch).sum().item()
        total_count += len(y_batch)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def predict_all(model, loader, device):
    model.eval()
    all_preds = []
    all_true = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_true.extend(y_batch.cpu().numpy().tolist())
            total_loss += loss.item() * len(y_batch)
            total_count += len(y_batch)

    return np.asarray(all_true), np.asarray(all_preds), total_loss / max(total_count, 1)


def load_checkpoint_if_available(model, optimizer, model_path: str, device, class_names: List[str]) -> int:
    """Load shared checkpoint if possible. Returns starting epoch number."""
    if not os.path.exists(model_path):
        print("\nNo previous model found. Training from scratch.")
        return 0

    print("\nFound existing model:")
    print(model_path)
    print("Loading with weights_only=False because this is your own local checkpoint file.")

    try:
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            old_classes = checkpoint.get("class_names", None)
            if old_classes is not None and list(old_classes) != list(class_names):
                print("Previous model has different classes. Starting from scratch.")
                print("Previous classes:", old_classes)
                print("Current classes: ", class_names)
                return 0

            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                except Exception:
                    print("Loaded model weights, but optimizer state was not compatible. Continuing with a fresh optimizer.")
            start_epoch = int(checkpoint.get("epoch", 0))
            print("Loaded existing CNN checkpoint successfully.")
            print("Training will continue from the previous model.")
            return start_epoch

        # Older style: checkpoint is directly a state_dict.
        model.load_state_dict(checkpoint)
        print("Loaded existing CNN weights successfully.")
        return 0

    except Exception as exc:
        print("\nCould not load the previous model into this V2 architecture.")
        print("Reason:", str(exc))
        print("Training will start from scratch for this run.")
        return 0


# ============================================================
# Result plots
# ============================================================

def plot_training_history(history: Dict[str, List[float]]) -> None:
    if not history["train_loss"]:
        return

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_acc, ax_loss, ax_err, ax_gap = axes.ravel()

    train_acc = np.array(history["train_acc"]) * 100
    val_acc = np.array(history["val_acc"]) * 100
    train_err = 100 - train_acc
    val_err = 100 - val_acc

    ax_acc.plot(epochs, train_acc, marker="o", label="Training accuracy")
    ax_acc.plot(epochs, val_acc, marker="o", label="Validation accuracy")
    ax_acc.set_title("Accuracy progress")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True)
    ax_acc.legend()

    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    ax_loss.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    ax_loss.set_title("Loss progress")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)
    ax_loss.legend()

    ax_err.plot(epochs, train_err, marker="o", label="Training error")
    ax_err.plot(epochs, val_err, marker="o", label="Validation error")
    ax_err.set_title("Error rate progress")
    ax_err.set_xlabel("Epoch")
    ax_err.set_ylabel("Error rate (%)")
    ax_err.grid(True)
    ax_err.legend()

    ax_gap.plot(epochs, val_acc - train_acc, marker="o")
    ax_gap.axhline(0, linewidth=1)
    ax_gap.set_title("Validation minus training accuracy")
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("Accuracy gap (%)")
    ax_gap.grid(True)

    fig.suptitle("CNN Training Dashboard", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> None:
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy(cm: np.ndarray, class_names: List[str]) -> None:
    correct = np.diag(cm)
    totals = cm.sum(axis=1)
    per_class_acc = np.divide(correct, totals, out=np.zeros_like(correct, dtype=float), where=totals != 0) * 100

    plt.figure(figsize=(9, 5))
    plt.bar(class_names, per_class_acc)
    plt.title("Per-class accuracy")
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="POF/RF OFDM modulation classifier with grouped RF plots")
    parser.add_argument("--data", required=True, help="Path to extracted dataset folder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--max-windows-per-dataset", type=int, default=100)
    parser.add_argument("--skip-signal-plots", action="store_true", help="Skip RF signal-analysis plots")
    parser.add_argument("--max-classes-to-plot", type=int, default=6, help="Number of representative classes to plot")
    parser.add_argument("--preview-samples", type=int, default=65536, help="Samples used only for RF preview plots")
    parser.add_argument("--no-resume", action="store_true", help="Do not load previous ML/Models/iq_cnn_classifier.pt")
    args = parser.parse_args()

    start_time = time.time()

    print_header("POF / RF Deep Learning Classifier V2")
    print("Dataset path:", args.data)
    print("Epochs:", args.epochs)
    print("Window size:", args.window_size)
    print("Max windows per H5 dataset:", args.max_windows_per_dataset)
    print("Shared model path:", MODEL_PATH)
    print("Plots are shown on screen only. PNG files are not saved.")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset path does not exist: {args.data}")

    print_header("Step 1: Scanning Dataset")
    records = find_h5_files(args.data)
    if not records:
        raise RuntimeError("No .h5 files found. Check that the .7z dataset was extracted.")

    class_names = sorted(list({r[1] for r in records}))
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    print(f"Detected H5 files: {len(records)}")
    print(f"Detected classes: {class_names}")

    files_per_class = Counter([r[1] for r in records])
    print("\nFiles per class:")
    for label in class_names:
        print(f"  {label}: {files_per_class[label]} files")

    snrs = sorted([r[3] for r in records if r[3] is not None])
    if snrs:
        print(f"\nDetected SNR range: {min(snrs)} dB to {max(snrs)} dB")

    if not args.skip_signal_plots:
        plot_rf_analysis(records, max_classes=args.max_classes_to_plot, preview_samples=args.preview_samples)

    X, y = build_window_dataset(
        records=records,
        label_to_idx=label_to_idx,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_dataset=args.max_windows_per_dataset,
    )

    print_header("Step 3: Train / Validation / Test Split")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    print(f"Training windows:   {len(y_train)}")
    print(f"Validation windows: {len(y_val)}")
    print(f"Test windows:       {len(y_test)}")

    train_loader = DataLoader(IQWindowDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(IQWindowDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(IQWindowDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nUsing device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("CPU training is slower, but fine for smaller tests.")

    print_header("Step 4: Building / Loading CNN")
    model = IQCNN(num_classes=len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    if args.no_resume:
        print("Resume disabled. Training from scratch.")
    else:
        start_epoch = load_checkpoint_if_available(model, optimizer, MODEL_PATH, device, class_names)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None

    print_header("Step 5: Training CNN")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, optimizer, device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        displayed_epoch = start_epoch + epoch
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"(overall {displayed_epoch}) | "
            f"train acc {train_acc*100:6.2f}% | "
            f"val acc {val_acc*100:6.2f}% | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_names": class_names,
                "window_size": args.window_size,
                "epoch": displayed_epoch,
                "best_val_acc": best_val_acc,
                "model_name": "IQCNN",
                "script": "ML_POF_V2.py",
            }

    print_header("Step 6: Saving Best CNN Model")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
        model.load_state_dict(best_state["model_state_dict"])
        print(f"Best model saved to: {MODEL_PATH}")
        print(f"Best validation accuracy this run: {best_val_acc*100:.2f}%")
    else:
        print("No model was saved because no training state was created.")

    print_header("Step 7: Testing Best Model")
    y_true, y_pred, test_loss = predict_all(model, test_loader, device)
    test_acc = accuracy_score(y_true, y_pred)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc*100:.2f}%")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:")
    print(cm)

    print_header("Step 8: Training and Evaluation Graphs")
    plot_training_history(history)
    plot_confusion_matrix(cm, class_names)
    plot_per_class_accuracy(cm, class_names)

    elapsed = time.time() - start_time
    print_header("Training Summary")
    print(f"Classes: {len(class_names)}")
    print(f"Total windows used: {len(y)}")
    print(f"Best validation accuracy this run: {best_val_acc*100:.2f}%")
    print(f"Final test accuracy: {test_acc*100:.2f}%")
    print(f"Shared CNN model saved at: {MODEL_PATH}")
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
    print("\nDone.")


if __name__ == "__main__":
    main()
