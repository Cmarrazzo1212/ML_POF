"""
User-friendly OFDM / RF / POF IQ Signal Classifier
--------------------------------------------------
This version reduces noisy output and creates plots:
  - example IQ waveform
  - example constellation scatter plot
  - example autocorrelation plot
  - training/validation accuracy
  - training/validation error rate
  - training/validation loss
  - confusion matrix
  - per-class accuracy bar chart

For Spyder command-line options, use for a small test:
--data "C:/Users/usuario/Desktop/POF_research/OFDM Modulation Classification Dataset" --epochs 10 --max-windows-per-dataset 300 --window-size 2048
"""

import argparse
import os
import re
import zipfile
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -----------------------------
# Data utilities
# -----------------------------

def extract_zip_if_needed(data_path: str) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    p = Path(data_path)
    if p.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(p, "r") as z:
            z.extractall(tmp.name)
        return tmp.name, tmp
    return data_path, None


def find_h5_files(data_path: str) -> List[Path]:
    p = Path(data_path)
    if p.is_file() and p.suffix.lower() in [".h5", ".hdf5"]:
        return [p]
    return sorted(list(p.rglob("*.h5")) + list(p.rglob("*.hdf5")))


def clean_label(text: str) -> str:
    label = Path(text).stem
    label = re.sub(r"[-_]?\d+(\.\d+)?\s*dB", "", label, flags=re.IGNORECASE)
    label = label.strip("_- ")
    return label if label else Path(text).stem


def list_h5_datasets(h5_path: Path) -> List[str]:
    names = []
    with h5py.File(h5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(name)
        f.visititems(visitor)
    return names


def read_complex_dataset(h5_path: Path, dataset_name: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        x = f[dataset_name][()]
    x = np.asarray(x).squeeze()

    if np.iscomplexobj(x):
        return x.astype(np.complex64).reshape(-1)

    if x.ndim >= 2 and x.shape[-1] == 2:
        return (x[..., 0] + 1j * x[..., 1]).astype(np.complex64).reshape(-1)

    return x.astype(np.float32).reshape(-1).astype(np.complex64)


def get_snr_from_filename(path: Path) -> Optional[float]:
    """Extract SNR value from names like -10dB.h5, 0dB.h5, 18dB.h5."""
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*dB", path.name, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def find_example_signal(h5_files: List[Path]) -> Tuple[Optional[np.ndarray], Optional[str], Optional[str]]:
    """Read one example signal for visualization before training."""
    if not h5_files:
        return None, None, None
    # Prefer a moderate SNR file because it usually looks cleaner than very low SNR.
    scored = []
    for path in h5_files:
        snr = get_snr_from_filename(path)
        score = abs((snr if snr is not None else 0) - 10)
        scored.append((score, path))
    scored.sort(key=lambda x: x[0])

    for _, path in scored:
        datasets = list_h5_datasets(path)
        if datasets:
            name = datasets[0]
            return read_complex_dataset(path, name), clean_label(name), path.name
    return None, None, None


def iq_to_two_channels(iq: np.ndarray) -> np.ndarray:
    return np.stack([iq.real, iq.imag], axis=0).astype(np.float32)


def normalize_window(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / (std + eps)


def build_windows(
    h5_files: List[Path],
    window_size: int,
    stride: int,
    max_windows_per_dataset: Optional[int],
    label_from: str = "dataset",
    quiet: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X, y = [], []
    counts = {}
    skipped = 0

    print("\nStep 2: Loading IQ signal windows")
    print("This may take a little while. Individual file names are hidden to reduce clutter.\n")

    for h5_path in tqdm(h5_files, desc="Reading H5 files"):
        datasets = list_h5_datasets(h5_path)
        if not datasets:
            skipped += 1
            continue

        for ds_name in datasets:
            if label_from == "file":
                label = clean_label(h5_path.name)
            elif label_from == "parent":
                label = clean_label(h5_path.parent.name)
            else:
                label = clean_label(ds_name)

            iq = read_complex_dataset(h5_path, ds_name)
            n = len(iq)
            if n < window_size:
                skipped += 1
                continue

            starts = np.arange(0, n - window_size + 1, stride)
            if max_windows_per_dataset is not None and len(starts) > max_windows_per_dataset:
                rng = np.random.default_rng(42)
                starts = rng.choice(starts, size=max_windows_per_dataset, replace=False)
                starts = np.sort(starts)

            for s in starts:
                win = iq[s:s + window_size]
                win = normalize_window(iq_to_two_channels(win))
                X.append(win)
                y.append(label)

            counts[label] = counts.get(label, 0) + len(starts)

    if not X:
        raise RuntimeError("No valid windows found. Check your data path and HDF5 structure.")

    if skipped:
        print(f"Skipped {skipped} file/dataset entries because they were empty or too short.")

    return np.stack(X), np.asarray(y), counts


# -----------------------------
# PyTorch dataset/model
# -----------------------------

class IQDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
            all_true.extend(yb.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_true), np.array(all_pred)


# -----------------------------
# Friendly plots and text
# -----------------------------

def print_clean_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def plot_signal_examples(iq: Optional[np.ndarray], label: Optional[str], file_name: Optional[str], out_prefix: str, sample_count: int = 4096):
    """Create visual checks for one signal: waveform, constellation, autocorrelation."""
    if iq is None or len(iq) == 0:
        print("No example signal available for IQ/autocorrelation plots.")
        return

    iq = np.asarray(iq).reshape(-1)
    n = min(sample_count, len(iq))
    segment = iq[:n]
    title_extra = f"{label} from {file_name}" if label and file_name else "example signal"

    # 1) IQ waveform: real and imaginary components over sample index.
    plt.figure(figsize=(10, 5))
    plt.plot(np.real(segment), label="Real / I")
    plt.plot(np.imag(segment), label="Imaginary / Q", alpha=0.8)
    plt.xlabel("Sample index")
    plt.ylabel("Normalized amplitude")
    plt.title(f"Example IQ Waveform ({title_extra})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Constellation scatter: real vs imaginary.
    step = max(1, len(segment) // 2500)
    const_segment = segment[::step]
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(const_segment), np.imag(const_segment), s=8, alpha=0.45)
    plt.xlabel("Real / I")
    plt.ylabel("Imaginary / Q")
    plt.title(f"Example Constellation Scatter ({title_extra})")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # 3) Autocorrelation magnitude.
    centered = segment - np.mean(segment)
    corr = np.correlate(centered, centered, mode="full")
    corr = corr[len(corr) // 2:]
    corr_mag = np.abs(corr)
    corr_mag = corr_mag / (corr_mag[0] + 1e-12)
    max_lag = min(512, len(corr_mag))

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(max_lag), corr_mag[:max_lag])
    plt.xlabel("Lag")
    plt.ylabel("Normalized autocorrelation magnitude")
    plt.title(f"Autocorrelation of Example Signal ({title_extra})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_error_rate(history: Dict[str, List[float]], out_prefix: str):
    """Plot classification error rate, which is 1 - accuracy."""
    epochs = np.arange(1, len(history["train_acc"]) + 1)
    train_error = 1 - np.asarray(history["train_acc"])
    val_error = 1 - np.asarray(history["val_acc"])

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_error, marker="o", label="Training error rate")
    plt.plot(epochs, val_error, marker="o", label="Validation error rate")
    plt.xlabel("Epoch")
    plt.ylabel("Error rate")
    plt.title("Training Error Rate Over Time")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_learning_progress(history: Dict[str, List[float]], out_prefix: str):
    """Plot validation accuracy improvement per epoch."""
    epochs = np.arange(1, len(history["val_acc"]) + 1)
    val_acc_percent = 100 * np.asarray(history["val_acc"])
    best_so_far = np.maximum.accumulate(val_acc_percent)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, val_acc_percent, marker="o", label="Validation accuracy")
    plt.plot(epochs, best_so_far, marker="o", label="Best so far")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Learning Progress During Training")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_curves(history: Dict[str, List[float]], out_prefix: str):
    epochs = np.arange(1, len(history["train_acc"]) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_acc"], marker="o", label="Training accuracy")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Over Training")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model Loss Over Training")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_prefix: str):
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix: Actual Class vs Predicted Class")
    plt.colorbar(label="Number of test windows")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")

    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_per_class_accuracy(cm: np.ndarray, class_names: List[str], out_prefix: str):
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, per_class_acc)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy for Each Modulation Class")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return per_class_acc


def explain_results(test_acc: float, cm: np.ndarray, class_names: List[str]):
    per_class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    best_idx = int(np.argmax(per_class_acc))
    worst_idx = int(np.argmin(per_class_acc))

    print_clean_header("Plain-English Result Summary")
    print(f"Overall test accuracy: {test_acc * 100:.2f}%")
    print(f"Best recognized class:  {class_names[best_idx]} ({per_class_acc[best_idx] * 100:.2f}%)")
    print(f"Hardest class:          {class_names[worst_idx]} ({per_class_acc[worst_idx] * 100:.2f}%)")
    print("\nHow to read this:")
    print("- Accuracy near 16.7% means random guessing for 6 classes.")
    print("- Accuracy above 16.7% means the model is learning some signal patterns.")
    print("- If some classes are 0%, the model needs more data, more epochs, or a stronger architecture.")
    print("- The confusion matrix shows exactly which classes are being mixed up.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to folder, .h5, or .zip containing H5 files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows-per-dataset", type=int, default=300)
    parser.add_argument("--label-from", choices=["dataset", "file", "parent"], default="dataset")
    parser.add_argument("--model-out", default="iq_cnn_classifier.pt")
    parser.add_argument("--output-prefix", default="ofdm_results")
    args = parser.parse_args()

    print_clean_header("OFDM / RF / POF IQ Deep Learning Classifier")
    print("This script trains a CNN on complex IQ signal windows.")
    print("It hides noisy file-by-file output and creates readable research-style plots.")

    data_root, tmp = extract_zip_if_needed(args.data)

    try:
        print_clean_header("Step 1: Dataset Check")
        h5_files = find_h5_files(data_root)
        print(f"Dataset path: {data_root}")
        print(f"H5 files found: {len(h5_files)}")
        print(f"Window size: {args.window_size} IQ samples")
        print(f"Max windows per H5 dataset: {args.max_windows_per_dataset}")
        print(f"Epochs: {args.epochs}")

        if not h5_files:
            raise RuntimeError("No .h5 or .hdf5 files found.")

        print_clean_header("Step 1B: Signal Preview Graphs")
        example_iq, example_label, example_file = find_example_signal(h5_files)
        if example_iq is not None:
            print(f"Creating example signal plots from: {example_label} / {example_file}")
            plot_signal_examples(example_iq, example_label, example_file, args.output_prefix)
        else:
            print("No example signal could be opened for preview plots.")

        X, labels, counts = build_windows(
            h5_files=h5_files,
            window_size=args.window_size,
            stride=args.stride,
            max_windows_per_dataset=args.max_windows_per_dataset,
            label_from=args.label_from,
            quiet=True,
        )

        print_clean_header("Step 3: Class Summary")
        total_windows = len(labels)
        print(f"Total training windows created: {total_windows}")
        for k, v in sorted(counts.items()):
            print(f"{k:12s}: {v:6d} windows")

        le = LabelEncoder()
        y = le.fit_transform(labels)
        class_names = list(le.classes_)
        num_classes = len(class_names)

        print(f"\nDetected {num_classes} classes:")
        for i, name in enumerate(class_names):
            print(f"  {i}: {name}")

        if num_classes < 2:
            print("Only one class was detected, so classification training cannot continue.")
            return

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )

        print_clean_header("Step 4: Train / Validation / Test Split")
        print(f"Training windows:   {len(X_train)}")
        print(f"Validation windows: {len(X_val)}")
        print(f"Test windows:       {len(X_test)}")

        train_loader = DataLoader(IQDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(IQDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(IQDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_clean_header("Step 5: Training")
        print(f"Using device: {device}")
        if str(device) == "cpu":
            print("CPU training is slower. This is okay for testing.")

        model = IQCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

        best_val_acc = 0.0
        best_state = None
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_acc)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:02d}/{args.epochs}"
                f"train acc: {train_acc * 100:6.2f}% | "
                f"val acc: {val_acc * 100:6.2f}% | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "window_size": args.window_size,
                    "label_from": args.label_from,
                }
                
                import os

                os.makedirs("ML/Models", exist_ok=True)

        if best_state is not None:
            torch.save(best_state,"ML/Models/iq_cnn_classifier.pt")
            model.load_state_dict(best_state["model_state_dict"])
            print(f"\nBest model saved to: {args.model_out}")

        print_clean_header("Step 6: Final Test Results")
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        cm = confusion_matrix(y_true, y_pred)

        print(f"Test accuracy: {test_acc * 100:.2f}%")
        print(f"Test loss: {test_loss:.4f}")

        print("\nDetailed class report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

        explain_results(test_acc, cm, class_names)

        print_clean_header("Step 7: Creating Graphs")
        plot_training_curves(history, args.output_prefix)
        plot_error_rate(history, args.output_prefix)
        plot_learning_progress(history, args.output_prefix)
        plot_confusion_matrix(cm, class_names, args.output_prefix)
        plot_per_class_accuracy(cm, class_names, args.output_prefix)

        print("Saved graph files:")
        print(f"  {args.output_prefix}_example_iq_waveform.png")
        print(f"  {args.output_prefix}_example_constellation.png")
        print(f"  {args.output_prefix}_example_autocorrelation.png")
        print(f"  {args.output_prefix}_accuracy.png")
        print(f"  {args.output_prefix}_loss.png")
        print(f"  {args.output_prefix}_error_rate.png")
        print(f"  {args.output_prefix}_learning_progress.png")
        print(f"  {args.output_prefix}_confusion_matrix.png")
        print(f"  {args.output_prefix}_per_class_accuracy.png")
        print("\nDone.")

    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
