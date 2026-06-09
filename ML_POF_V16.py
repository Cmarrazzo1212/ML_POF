# -*- coding: utf-8 -*-
r"""
ML_POF_V16.py

Universal Machine Learning pipeline for Plastic Optical Fiber (POF), OFDM,
ADALM-Pluto SDR, MATLAB, Python, HDF5, CSV/TXT, NumPy, and other physical-signal data.
im 

Default save locations on Windows:
    Script: C:\Users\usuario\Desktop\POF_research\ML_POF_V13.py
    Model:  C:\Users\usuario\Desktop\POF_research\ML\Models\memory.pt

Main features
-------------
1. Reads many input formats:
   - .h5 / .hdf5: complex datasets, X/Y datasets, nested datasets
   - .mat: MATLAB variables, X/Y variables, complex arrays
   - .npy / .npz: NumPy arrays or X/Y archives
   - .csv / .txt: real-only, I/Q columns, or label column
   - .wav: optional, if scipy is installed
2. Converts long recordings into fixed windows: (N, 2, WINDOW_SIZE)
3. Learns labels from:
   - Y / y / labels arrays inside files
   - dataset names such as BPSK_QPSK0dB
   - folder names such as BPSK_QPSK
   - interactive/manual fallback if needed
4. Trains either:
   - TRNN residual model for IQ modulation / POF signal classification
   - Simple 1D CNN for faster tests
5. Saves/resumes a checkpoint named memory.pt.
6. Produces user-friendly plots:
   - signal overview
   - constellation
   - spectrum / PSD
   - spectrogram
   - amplitude and phase histograms
   - training curves
   - confusion matrix
   - per-class accuracy
   - prediction-confidence plot
7. Has three ways to run:
   - no arguments: opens a Tkinter folder/file picker if possible
   - command line / Spyder args
   - edit USER SETTINGS below

Quick first test in Spyder or terminal:
    python ML_POF_V13.py --data "C:/Users/usuario/Desktop/POF_research/OFDM Modulation Classification Dataset" --epochs 3 --max-windows-per-file 50

For your uploaded example H5 format, a folder can contain files like:
    0dB.h5   with dataset BPSK_QPSK0dB shaped (4194304, 1) complex64

For future MATLAB / ADALM-Pluto captures, save .mat with any of these patterns:
    X, Y
    iq, label
    rx, label
    signal, label
    any complex vector/matrix where the file/folder/variable name contains the class
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Helps Spyder/Windows avoid OpenMP duplicate-library crashes sometimes seen with torch/sklearn.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np

import matplotlib.pyplot as plt

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

try:
    from scipy.io import wavfile
except Exception:
    wavfile = None

try:
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
except Exception as exc:
    raise ImportError("This script needs scikit-learn. Install with: pip install scikit-learn") from exc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# 0. USER SETTINGS
# =============================================================================

PROJECT_DIR = Path(r"C:\Users\usuario\Desktop\POF_research")
DEFAULT_MODEL_DIR = PROJECT_DIR / "ML" / "Models"
DEFAULT_MEMORY_NAME = "memory.pt"
DEFAULT_DATA_DIR = PROJECT_DIR
DEFAULT_PLOTS_DIR = PROJECT_DIR / "ML" / "Plots"

SUPPORTED_EXTENSIONS = {
    ".h5", ".hdf5", ".mat", ".npz", ".npy", ".csv", ".txt", ".wav"
}

DEFAULT_CLASS_NAMES = [
    "BPSK+BPSK", "BPSK+QPSK", "BPSK+8PSK",
    "QPSK+BPSK", "QPSK+QPSK", "QPSK+8PSK",
]

# Known aliases from your older work. You can add POF experimental labels later.
DEFAULT_LABEL_ALIASES = {
    "BPSK_BPSK": "BPSK+BPSK", "BPSK+BPSK": "BPSK+BPSK", "BPSKBPSK": "BPSK+BPSK",
    "BPSK_QPSK": "BPSK+QPSK", "BPSK+QPSK": "BPSK+QPSK", "BPSKQPSK": "BPSK+QPSK",
    "BPSK_8PSK": "BPSK+8PSK", "BPSK+8PSK": "BPSK+8PSK", "BPSK8PSK": "BPSK+8PSK",
    "QPSK_BPSK": "QPSK+BPSK", "QPSK+BPSK": "QPSK+BPSK", "QPSKBPSK": "QPSK+BPSK",
    "QPSK_QPSK": "QPSK+QPSK", "QPSK+QPSK": "QPSK+QPSK", "QPSKQPSK": "QPSK+QPSK",
    "QPSK_8PSK": "QPSK+8PSK", "QPSK+8PSK": "QPSK+8PSK", "QPSK8PSK": "QPSK+8PSK",
}


# =============================================================================
# 1. CONFIG / SMALL HELPERS
# =============================================================================

@dataclass
class Config:
    data: str = str(DEFAULT_DATA_DIR)
    memory_path: str = str(DEFAULT_MODEL_DIR / DEFAULT_MEMORY_NAME)
    plots_dir: str = str(DEFAULT_PLOTS_DIR)
    window_size: int = 1024
    stride: int = 1024
    max_windows_per_file: Optional[int] = 200
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-4
    validation_split: float = 0.20
    seed: int = 42
    model: str = "trnn"  # trnn or cnn
    save_plots: bool = False
    show_plots: bool = True
    no_train: bool = False
    prediction_file: Optional[str] = None
    allow_unknown_labels: bool = False
    preview_samples: int = 32768
    device: str = "auto"


def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalize_text(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(request: str) -> torch.device:
    if request.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_label_text(text: str) -> str:
    """Turns names like BPSK_QPSK0dB into BPSK+QPSK when possible."""
    raw = str(text)
    raw = re.sub(r"[-_ ]?-?\d+(?:\.\d+)?\s*dB", "", raw, flags=re.IGNORECASE)
    raw = raw.strip("_- /\\")
    norm = normalize_text(raw)
    for alias, label in DEFAULT_LABEL_ALIASES.items():
        if normalize_text(alias) in norm or normalize_text(alias).replace("_", "") in norm.replace("_", ""):
            return label
    return raw if raw else str(text)


def infer_label_from_text(text: str) -> Optional[str]:
    norm = normalize_text(text)
    compact = norm.replace("_", "")
    for alias, label in DEFAULT_LABEL_ALIASES.items():
        a = normalize_text(alias)
        if a in norm or a.replace("_", "") in compact:
            return label
    # Generic fallback: if the name is not just dB or data-like, use it after cleaning.
    cleaned = clean_label_text(text)
    reject = {"X", "Y", "DATA", "SIGNAL", "IQ", "RX", "TX", "LABEL", "LABELS", "SNR", "SNRDB"}
    if normalize_text(cleaned) and normalize_text(cleaned) not in reject and not re.fullmatch(r"-?\d+(\.\d+)?DB", normalize_text(str(text))):
        return cleaned
    return None


def infer_label_from_path(path: Path, variable_name: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if variable_name:
        candidates.append(variable_name)
    candidates.append(path.stem)
    candidates.extend(parent.name for parent in list(path.parents)[:5])
    for candidate in candidates:
        label = infer_label_from_text(candidate)
        if label:
            return label
    return None


def extract_snr(text: str) -> Optional[float]:
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*dB", str(text), flags=re.IGNORECASE)
    return float(m.group(1)) if m else None


# =============================================================================
# 2. DATA CONVERSION
# =============================================================================

def make_complex_array(x: np.ndarray) -> np.ndarray:
    """Accepts complex, real-only, or I/Q-column data and returns a 1D complex vector."""
    arr = np.asarray(x).squeeze()
    if arr.size == 0:
        raise ValueError("Empty array")
    if np.iscomplexobj(arr):
        return arr.astype(np.complex64).reshape(-1)
    if arr.ndim == 2 and 2 in arr.shape:
        if arr.shape[0] == 2 and arr.shape[1] != 2:
            return (arr[0, :] + 1j * arr[1, :]).astype(np.complex64).reshape(-1)
        return (arr[:, 0] + 1j * arr[:, 1]).astype(np.complex64).reshape(-1)
    return arr.astype(np.float32).reshape(-1).astype(np.complex64)


def normalize_complex_window(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg, dtype=np.complex64).reshape(-1)
    seg = seg - np.mean(seg)
    power = np.sqrt(np.mean(np.abs(seg) ** 2))
    if power > 0:
        seg = seg / (power + 1e-8)
    return np.stack([seg.real, seg.imag], axis=0).astype(np.float32)


def windows_from_signal(signal: np.ndarray, window_size: int, stride: int, max_windows: Optional[int]) -> np.ndarray:
    sig = make_complex_array(signal)
    if len(sig) < window_size:
        raise ValueError(f"Signal too short: {len(sig)} samples; need at least {window_size}.")
    starts = np.arange(0, len(sig) - window_size + 1, stride, dtype=np.int64)
    if max_windows is not None and len(starts) > max_windows:
        rng = np.random.default_rng(42)
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))
    return np.stack([normalize_complex_window(sig[s:s + window_size]) for s in starts], axis=0)


def ensure_windowed_x(x: np.ndarray, cfg: Config) -> np.ndarray:
    """Converts common formats into X shape (N, 2, window_size)."""
    arr = np.asarray(x).squeeze()
    W = cfg.window_size

    # Already windowed: (N, 2, W)
    if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == W:
        return arr.astype(np.float32)

    # Already windowed: (N, W, 2)
    if arr.ndim == 3 and arr.shape[1] == W and arr.shape[2] == 2:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)

    # Already windowed complex: (N, W)
    if np.iscomplexobj(arr) and arr.ndim == 2 and arr.shape[1] == W:
        return np.stack([arr.real, arr.imag], axis=1).astype(np.float32)

    # Matrix of I/Q columns or any long vector becomes windows.
    return windows_from_signal(arr, cfg.window_size, cfg.stride, cfg.max_windows_per_file)



def safe_h5_array(dset):
    """
    Read an HDF5 dataset safely.

    V11 rule: avoid fancy/ambiguous h5py slicing. Some MATLAB-created H5 files
    can throw: "'<' not supported between instances of 'slice' and 'int'"
    when read with generic slicing. This function handles common signal shapes
    explicitly and skips unsupported MATLAB reference/object datasets cleanly.
    """
    if not isinstance(dset, h5py.Dataset):
        return np.asarray(dset)

    # MATLAB v7.3 files may contain object/reference datasets that are not signals.
    if dset.dtype.kind == "O":
        raise ValueError(f"Skipping MATLAB/object-reference dataset: shape={dset.shape}, dtype={dset.dtype}")

    shape = tuple(int(v) for v in dset.shape)

    # Scalar datasets.
    if len(shape) == 0:
        return np.asarray(dset[()])

    # Most reliable full read.
    try:
        return np.asarray(dset[()])
    except Exception as first_exc:
        # Explicit common fallbacks.
        try:
            if len(shape) == 1:
                return np.asarray(dset[0:shape[0]])
            if len(shape) == 2:
                r, c = shape
                if c == 1:
                    return np.asarray(dset[0:r, 0])
                if r == 1:
                    return np.asarray(dset[0, 0:c])
                return np.asarray(dset[0:r, 0:c])
            if len(shape) == 3:
                return np.asarray(dset[0:shape[0], 0:shape[1], 0:shape[2]])
        except Exception as second_exc:
            raise RuntimeError(
                f"Could not read H5 dataset safely. shape={shape}, dtype={dset.dtype}. "
                f"First error={first_exc}. Second error={second_exc}."
            )
        raise RuntimeError(f"Could not read H5 dataset safely. shape={shape}, dtype={dset.dtype}. Error={first_exc}")

# =============================================================================
# 3. FILE LOADERS
# =============================================================================

@dataclass
class LoadedBlock:
    X: np.ndarray
    y_text: List[str]
    snr: np.ndarray
    source: str


def h5_dataset_names(h5: h5py.Group, prefix: str = "") -> List[str]:
    names: List[str] = []
    for key in h5.keys():
        obj = h5[key]
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(obj, h5py.Dataset):
            names.append(name)
        elif isinstance(obj, h5py.Group):
            names.extend(h5_dataset_names(obj, name))
    return names


def load_h5(path: Path, cfg: Config) -> List[LoadedBlock]:
    blocks: List[LoadedBlock] = []
    with h5py.File(path, "r") as f:
        keys = h5_dataset_names(f)
        lower = {k.lower().split("/")[-1]: k for k in keys}
        if not keys:
            return blocks
        print(f"  H5 datasets found: {[(k, tuple(f[k].shape), str(f[k].dtype)) for k in keys[:8]]}")

        # Case 1: structured X/Y file.
        if "x" in lower:
            X = ensure_windowed_x(safe_h5_array(f[lower["x"]]), cfg)
            y_key = lower.get("y") or lower.get("label") or lower.get("labels")
            if y_key:
                y_raw = np.asarray(safe_h5_array(f[y_key])).squeeze().reshape(-1)
                if len(y_raw) == 1:
                    y_text = [str(int(y_raw[0]))] * len(X)
                elif len(y_raw) == len(X):
                    y_text = [str(v) for v in y_raw]
                else:
                    y_text = [infer_label_from_path(path) or "unknown"] * len(X)
            else:
                label = infer_label_from_path(path) or "unknown"
                y_text = [label] * len(X)
            snr_key = lower.get("snrdb") or lower.get("snr")
            snr_val = float(np.asarray(safe_h5_array(f[snr_key])).squeeze().reshape(-1)[0]) if snr_key else extract_snr(str(path))
            snr = np.full(len(X), np.nan if snr_val is None else snr_val, dtype=np.float32)
            blocks.append(LoadedBlock(X, y_text, snr, str(path)))
            return blocks

        # Case 2: every useful dataset is a long signal; label from dataset/path.
        for key in keys:
            base = key.lower().split("/")[-1]
            if base in {"y", "label", "labels", "snr", "snrdb"}:
                continue
            try:
                raw = safe_h5_array(f[key])
                X = ensure_windowed_x(raw, cfg)
            except Exception as exc:
                print(f"  Skipping H5 dataset {key}: {exc}")
                continue
            label = infer_label_from_path(path, key) or infer_label_from_path(path) or "unknown"
            snr_val = extract_snr(key) if extract_snr(key) is not None else extract_snr(str(path))
            snr = np.full(len(X), np.nan if snr_val is None else snr_val, dtype=np.float32)
            blocks.append(LoadedBlock(X, [label] * len(X), snr, f"{path}::{key}"))
    return blocks


def load_mat(path: Path, cfg: Config) -> List[LoadedBlock]:
    if loadmat is None:
        raise ImportError("scipy is needed for .mat files. Install with: pip install scipy")
    mat = loadmat(path)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    lower = {k.lower(): k for k in keys}
    blocks: List[LoadedBlock] = []
    if "x" in lower:
        X = ensure_windowed_x(mat[lower["x"]], cfg)
        y_key = lower.get("y") or lower.get("label") or lower.get("labels")
        if y_key:
            y_raw = np.asarray(mat[y_key]).squeeze().reshape(-1)
            y_text = [str(v) for v in (np.repeat(y_raw, len(X)) if len(y_raw) == 1 else y_raw[:len(X)])]
        else:
            y_text = [infer_label_from_path(path) or "unknown"] * len(X)
        snr_val = extract_snr(str(path))
        blocks.append(LoadedBlock(X, y_text, np.full(len(X), np.nan if snr_val is None else snr_val), str(path)))
        return blocks
    for key in keys:
        arr = np.asarray(mat[key])
        if arr.size < cfg.window_size:
            continue
        try:
            X = ensure_windowed_x(arr, cfg)
        except Exception:
            continue
        label = infer_label_from_path(path, key) or "unknown"
        snr_val = extract_snr(key) if extract_snr(key) is not None else extract_snr(str(path))
        blocks.append(LoadedBlock(X, [label] * len(X), np.full(len(X), np.nan if snr_val is None else snr_val), f"{path}::{key}"))
    return blocks


def load_np(path: Path, cfg: Config) -> List[LoadedBlock]:
    blocks: List[LoadedBlock] = []
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        lower = {k.lower(): k for k in keys}
        if "x" in lower:
            X = ensure_windowed_x(data[lower["x"]], cfg)
            y_key = lower.get("y") or lower.get("label") or lower.get("labels")
            label = infer_label_from_path(path) or "unknown"
            y_text = [label] * len(X) if not y_key else [str(v) for v in np.asarray(data[y_key]).squeeze().reshape(-1)[:len(X)]]
            snr_val = extract_snr(str(path))
            blocks.append(LoadedBlock(X, y_text, np.full(len(X), np.nan if snr_val is None else snr_val), str(path)))
            return blocks
        for key in keys:
            try:
                X = ensure_windowed_x(data[key], cfg)
            except Exception:
                continue
            label = infer_label_from_path(path, key) or "unknown"
            snr_val = extract_snr(key) if extract_snr(key) is not None else extract_snr(str(path))
            blocks.append(LoadedBlock(X, [label] * len(X), np.full(len(X), np.nan if snr_val is None else snr_val), f"{path}::{key}"))
    else:
        arr = np.load(path, allow_pickle=True)
        X = ensure_windowed_x(arr, cfg)
        label = infer_label_from_path(path) or "unknown"
        snr_val = extract_snr(str(path))
        blocks.append(LoadedBlock(X, [label] * len(X), np.full(len(X), np.nan if snr_val is None else snr_val), str(path)))
    return blocks


def load_csv_txt(path: Path, cfg: Config) -> List[LoadedBlock]:
    # Try comma first, then whitespace.
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
        if arr.dtype.names:
            names = list(arr.dtype.names)
            numeric_cols = []
            label_values = None
            for name in names:
                col = arr[name]
                if normalize_text(name) in {"LABEL", "Y", "CLASS", "TARGET"}:
                    label_values = [str(v) for v in col]
                elif np.issubdtype(np.asarray(col).dtype, np.number):
                    numeric_cols.append(np.asarray(col, dtype=np.float32))
            data = np.vstack(numeric_cols).T if numeric_cols else np.asarray([])
            X = ensure_windowed_x(data, cfg)
            label = infer_label_from_path(path) or "unknown"
            y_text = [label] * len(X) if label_values is None else [label_values[0]] * len(X)
        else:
            raise ValueError
    except Exception:
        try:
            data = np.loadtxt(path, delimiter=",")
        except Exception:
            data = np.loadtxt(path)
        X = ensure_windowed_x(data, cfg)
        y_text = [infer_label_from_path(path) or "unknown"] * len(X)
    snr_val = extract_snr(str(path))
    return [LoadedBlock(X, y_text, np.full(len(X), np.nan if snr_val is None else snr_val), str(path))]


def load_wav(path: Path, cfg: Config) -> List[LoadedBlock]:
    if wavfile is None:
        raise ImportError("scipy is needed for .wav files. Install with: pip install scipy")
    rate, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim == 2 and data.shape[1] >= 2:
        data = data[:, :2]
    X = ensure_windowed_x(data, cfg)
    label = infer_label_from_path(path) or "unknown"
    snr_val = extract_snr(str(path))
    return [LoadedBlock(X, [label] * len(X), np.full(len(X), np.nan if snr_val is None else snr_val), f"{path} @ {rate}Hz")]


def load_one_file(path: Path, cfg: Config) -> List[LoadedBlock]:
    ext = path.suffix.lower()
    if ext in {".h5", ".hdf5"}:
        return load_h5(path, cfg)
    if ext == ".mat":
        return load_mat(path, cfg)
    if ext in {".npz", ".npy"}:
        return load_np(path, cfg)
    if ext in {".csv", ".txt"}:
        return load_csv_txt(path, cfg)
    if ext == ".wav":
        return load_wav(path, cfg)
    return []


def discover_files(data_path: Path) -> List[Path]:
    if data_path.is_file() and data_path.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [data_path]
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    files = [p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def build_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    print_header("Step 1: Loading signal data")
    data_path = Path(cfg.data)
    files = discover_files(data_path)
    print(f"Data path: {data_path}")
    print(f"Files found: {len(files)}")
    if not files:
        raise RuntimeError("No supported data files found.")

    X_parts: List[np.ndarray] = []
    labels: List[str] = []
    snr_parts: List[np.ndarray] = []
    sources: List[str] = []
    skipped: List[str] = []

    for i, file in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file.name}")
        try:
            blocks = load_one_file(file, cfg)
        except Exception as exc:
            skipped.append(f"{file}: {exc}")
            print(f"  Skipped: {exc}")
            if len(skipped) <= 3:
                import traceback
                print("  Debug traceback for first skipped files:")
                traceback.print_exc(limit=3)
            continue
        for block in blocks:
            if not cfg.allow_unknown_labels and any(str(v).lower() == "unknown" for v in block.y_text):
                skipped.append(f"{block.source}: unknown label")
                print(f"  Skipped block with unknown label: {block.source}")
                continue
            X_parts.append(block.X)
            labels.extend(block.y_text)
            snr_parts.append(block.snr.astype(np.float32))
            sources.append(block.source)
            print(f"  Loaded {len(block.X)} windows from {block.source}; label={block.y_text[0] if block.y_text else 'unknown'}")

    if not X_parts:
        raise RuntimeError(
            "No usable training windows were loaded. Make sure files contain labels in Y/y/labels, "
            "or put files in class folders/names like BPSK_QPSK, or use --allow-unknown-labels for inspection only."
        )

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    snr = np.concatenate(snr_parts, axis=0).astype(np.float32)

    # Stable class order: known labels first, then anything new alphabetically.
    unique_found = sorted(set(labels))
    class_names = [c for c in DEFAULT_CLASS_NAMES if c in unique_found]
    class_names += [c for c in unique_found if c not in class_names]
    label_to_id = {name: i for i, name in enumerate(class_names)}
    y = np.asarray([label_to_id[name] for name in labels], dtype=np.int64)

    meta = {
        "class_names": class_names,
        "label_to_id": label_to_id,
        "window_size": cfg.window_size,
        "stride": cfg.stride,
        "sources": sources,
        "skipped": skipped,
        "config": asdict(cfg),
    }

    print_header("Step 2: Dataset summary")
    print(f"X shape: {X.shape}  (windows, I/Q channels, samples)")
    print(f"y shape: {y.shape}")
    print(f"Classes: {class_names}")
    for cname in class_names:
        print(f"  {cname:20s}: {(y == label_to_id[cname]).sum()} windows")
    if skipped:
        print("\nSkipped items:")
        for item in skipped[:20]:
            print("  -", item)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more")
    return X, y, snr, class_names, meta


# =============================================================================
# 4. MODELS
# =============================================================================

class IQDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self) -> int:
        return len(self.y)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def _conv_bn_relu(in_ch: int, out_ch: int, kernel: Tuple[int, int], padding: Tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, num_conv: int):
        super().__init__()
        self.body = nn.Sequential(*[
            _conv_bn_relu(channels, channels, kernel=(3, 1), padding=(1, 0))
            for _ in range(num_conv)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TripleSkipResidualStack(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool_size: Tuple[int, int]):
        super().__init__()
        self.entry = _conv_bn_relu(in_ch, out_ch, kernel=(1, 1), padding=(0, 0))
        self.ru_a = ResidualUnit(out_ch, 1)
        self.ru_b = ResidualUnit(out_ch, 2)
        self.ru_c = ResidualUnit(out_ch, 3)
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=False) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.proj(x)
        out = self.ru_c(self.ru_b(self.ru_a(self.entry(x))))
        if skip.shape[2:] != out.shape[2:]:
            skip = F.adaptive_avg_pool2d(skip, out.shape[2:])
        return self.pool(out + skip)


class TRNN(nn.Module):
    _POOL_SIZES = [(2, 2)] + [(1, 2)] * 6
    def __init__(self, num_classes: int):
        super().__init__()
        blocks: List[nn.Module] = []
        in_ch = 1
        for pool in self._POOL_SIZES:
            blocks.append(TripleSkipResidualStack(in_ch, 32, pool))
            in_ch = 32
        self.trs = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 128),
            nn.SiLU(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 2, W) -> (B, 1, 2, W)
        return self.classifier(self.gap(self.trs(x)))


class SimpleIQCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.35), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.20), nn.Linear(128, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name.lower() == "cnn":
        return SimpleIQCNN(num_classes)
    if model_name.lower() == "trnn":
        return TRNN(num_classes)
    raise ValueError("model must be 'trnn' or 'cnn'")


# =============================================================================
# 5. TRAINING / EVALUATION
# =============================================================================

def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: Optional[optim.Optimizer] = None) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return total_loss / max(total, 1), correct / max(total, 1)


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_pred: List[int] = []
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            y_pred.extend(np.argmax(p, axis=1).tolist())
    return np.asarray(y_pred, dtype=np.int64), np.vstack(probs)


def load_checkpoint_if_possible(model: nn.Module, optimizer: optim.Optimizer, memory_path: Path, class_names: List[str], device: torch.device) -> Dict[str, Any]:
    if not memory_path.exists():
        return {"start_epoch": 1, "best_val_acc": 0.0, "history": {}}
    print(f"Loading memory checkpoint: {memory_path}")
    ckpt = torch.load(memory_path, map_location=device)
    old_classes = ckpt.get("class_names")
    old_model = ckpt.get("model_name")
    if old_classes != class_names:
        print("  Existing memory has different classes. Starting fresh model but keeping old file safe until a new best is saved.")
        return {"start_epoch": 1, "best_val_acc": 0.0, "history": {}}
    try:
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        return {
            "start_epoch": int(ckpt.get("epoch", 0)) + 1,
            "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
            "history": ckpt.get("history", {}),
        }
    except Exception as exc:
        print(f"  Could not load existing checkpoint weights: {exc}")
        return {"start_epoch": 1, "best_val_acc": 0.0, "history": {}}


def save_checkpoint(memory_path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, best_val_acc: float, history: Dict[str, List[float]], class_names: List[str], cfg: Config, meta: Dict[str, Any]) -> None:
    ensure_dir(memory_path.parent)
    torch.save({
        "version": "ML_POF_V16",
        "created_unix": time.time(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "model_name": cfg.model,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_names": class_names,
        "config": asdict(cfg),
        "history": history,
        "metadata": meta,
    }, memory_path)


def train_model(X: np.ndarray, y: np.ndarray, class_names: List[str], cfg: Config, meta: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, List[float]], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    print_header("Step 3: Training model")
    device = choose_device(cfg.device)
    print(f"Device: {device}")
    model = build_model(cfg.model, len(class_names)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    stratify = y if min(np.bincount(y)) >= 2 and len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.validation_split, random_state=cfg.seed, stratify=stratify
    )
    train_loader = DataLoader(IQDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(IQDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    memory_path = Path(cfg.memory_path)
    load_info = load_checkpoint_if_possible(model, optimizer, memory_path, class_names, device)
    history: Dict[str, List[float]] = {
        "train_loss": list(load_info.get("history", {}).get("train_loss", [])),
        "train_acc": list(load_info.get("history", {}).get("train_acc", [])),
        "val_loss": list(load_info.get("history", {}).get("val_loss", [])),
        "val_acc": list(load_info.get("history", {}).get("val_acc", [])),
    }
    best_val_acc = float(load_info.get("best_val_acc", 0.0))
    start_epoch = int(load_info.get("start_epoch", 1))

    print(f"Training windows: {len(X_train)} | Validation windows: {len(X_val)}")
    print(f"Checkpoint: {memory_path}")
    print(f"Starting epoch: {start_epoch}")
    print("\n Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Saved")
    print("-" * 67)

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, start_epoch + cfg.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, None)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        saved = ""
        # Protect the best model: save only if this run beats the saved best.
        # Exception: if memory.pt does not exist yet, save the first valid model.
        should_save = (not memory_path.exists()) or (va_acc > best_val_acc)
        if should_save:
            best_val_acc = va_acc
            save_checkpoint(memory_path, model, optimizer, epoch, best_val_acc, history, class_names, cfg, meta)
            saved = "YES - NEW BEST"
        print(f"{epoch:6d} | {tr_loss:10.4f} | {tr_acc:9.3f} | {va_loss:8.4f} | {va_acc:7.3f} | {saved}")
        last_epoch = epoch

    # Reload the best memory for final evaluation if possible.
    if memory_path.exists():
        ckpt = torch.load(memory_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
        except Exception:
            pass

    y_pred, probs = predict_all(model, val_loader, device)
    print_header("Step 4: Validation report")
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))
    return model, history, (y_val, y_pred, probs)


# =============================================================================
# 6. PLOTS
# =============================================================================

def save_or_show(fig: plt.Figure, name: str, cfg: Config) -> None:
    if cfg.save_plots:
        out = Path(cfg.plots_dir) / f"{name}.png"
        ensure_dir(out.parent)
        fig.savefig(out, dpi=160, bbox_inches="tight")
        print(f"Saved plot: {out}")
    if cfg.show_plots:
        plt.show()
    else:
        plt.close(fig)


def _plot_grid_by_class(
    examples: List[Tuple[int, str, np.ndarray]],
    plot_func,
    title: str,
    name: str,
    cfg: Config,
    max_cols: int = 3,
) -> None:
    """Create separated subplots instead of one cluttered overlay."""
    if not examples:
        return
    cols = min(max_cols, len(examples))
    rows = int(math.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (_, label, sig) in zip(axes, examples):
        ax.axis("on")
        plot_func(ax, sig, label)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_or_show(fig, name, cfg)


def plot_data_preview(X: np.ndarray, y: np.ndarray, snr: np.ndarray, class_names: List[str], cfg: Config) -> None:
    print_header("Step 2B: Signal preview plots")
    print("Plot saving is OFF by default in V14. Plots are displayed only unless --save-plots is used.")

    # Pick one example per class.
    indices = []
    for cid in sorted(np.unique(y)):
        idxs = np.where(y == cid)[0]
        if len(idxs):
            indices.append(int(idxs[0]))
    if not indices:
        return

    examples: List[Tuple[int, str, np.ndarray]] = []
    for idx in indices:
        label = class_names[int(y[idx])]
        sig = X[idx, 0] + 1j * X[idx, 1]
        examples.append((idx, label, sig))

    n_time = min(cfg.window_size, 1200)

    def plot_time(ax, sig, label):
        ax.plot(np.real(sig[:n_time]), linewidth=1.0)
        ax.set_title(f"Real / I component: {label}")
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Normalized amplitude")
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_time,
        "Separated real/I time-domain previews by class",
        "01_time_domain_separated",
        cfg,
    )

    # -------------------------------------------------------------------------
    # V16 constellation preview: synchronization-aware symbol constellation.
    #
    # This is still ONLY a preview/analysis change. Training data loading,
    # H5/MAT/NPY/CSV/TXT/WAV support, checkpoint loading, and model training
    # are not changed.
    #
    # What V16 does differently:
    #   1) Prefer a later window for two-part files, so BPSK_QPSK shows the
    #      second modulation instead of the first BPSK half.
    #   2) Try every samples-per-symbol timing offset.
    #   3) Estimate and remove residual frequency/phase drift using the Mth
    #      power method for the expected PSK order.
    #   4) Choose the offset with the tightest PSK phase clustering.
    # -------------------------------------------------------------------------
    def _expected_second_mod_order(label: str) -> int:
        text = str(label).upper().replace("+", "_").replace("-", "_")
        parts = [p for p in text.split("_") if p]
        mod = parts[-1] if parts else text
        if "8PSK" in mod:
            return 8
        if "QPSK" in mod or "4PSK" in mod:
            return 4
        if "BPSK" in mod or "2PSK" in mod:
            return 2
        return 4

    def _normalize_points(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.complex64).reshape(-1)
        z = z[np.isfinite(z.real) & np.isfinite(z.imag)]
        if len(z) == 0:
            return z
        z = z - np.mean(z)
        rms = np.sqrt(np.mean(np.abs(z) ** 2))
        if rms > 0:
            z = z / (rms + 1e-8)
        return z

    def _mpsk_cfo_phase_correct(z: np.ndarray, M: int) -> np.ndarray:
        z = _normalize_points(z)
        if len(z) < 16:
            return z
        n = np.arange(len(z), dtype=np.float64)
        zm = z ** M
        phase = np.unwrap(np.angle(zm))
        good = np.abs(zm) > np.percentile(np.abs(zm), 30)
        if np.count_nonzero(good) >= 16:
            slope, intercept = np.polyfit(n[good], phase[good], 1)
        else:
            slope, intercept = np.polyfit(n, phase, 1)
        correction = np.exp(-1j * (slope * n + intercept) / M)
        return _normalize_points(z * correction)

    def _psk_cluster_score(z: np.ndarray, M: int) -> float:
        z = _normalize_points(z)
        if len(z) < 16:
            return -1e9
        # Rotate once more so the Mth-power mean is close to angle 0.
        mean_m = np.mean(z ** M)
        if np.abs(mean_m) > 1e-8:
            z = z * np.exp(-1j * np.angle(mean_m) / M)
        phase_error = np.angle(z ** M) / M
        phase_tightness = float(np.mean(np.cos(M * phase_error)))
        radial = np.abs(z)
        radial_stability = -float(np.std(radial) / (np.mean(radial) + 1e-8))
        return phase_tightness + 0.20 * radial_stability

    def _best_symbol_points(sig: np.ndarray, label: str, default_sps: int = 8):
        M = _expected_second_mod_order(label)
        sps = int(default_sps)
        # Use the later part of the available window, because the class names are
        # BPSK_QPSK / BPSK_8PSK and the second modulation is the important one.
        sig = np.asarray(sig, dtype=np.complex64).reshape(-1)
        if len(sig) >= 512:
            sig = sig[len(sig)//2:]
        best = None
        for off in range(sps):
            pts = sig[off::sps]
            pts = pts[np.abs(pts) > np.percentile(np.abs(pts), 10)] if len(pts) > 32 else pts
            pts = _mpsk_cfo_phase_correct(pts, M)
            score = _psk_cluster_score(pts, M)
            if best is None or score > best[0]:
                best = (score, pts, sps, off, M)
        if best is None:
            return np.asarray([], dtype=np.complex64), sps, 0, M, -1e9
        score, pts, sps, off, M = best
        return pts, sps, off, M, score

    symbol_examples = []
    rng = np.random.default_rng(cfg.seed)
    print("\nV16 constellation analysis:")
    for cid in sorted(np.unique(y)):
        idxs = np.where(y == cid)[0]
        if len(idxs) == 0:
            continue
        label = class_names[int(cid)]
        # Use a later window to favor the second modulation in two-part captures.
        chosen = int(idxs[max(0, int(0.75 * (len(idxs) - 1)))])
        sig2 = X[chosen, 0] + 1j * X[chosen, 1]
        pts, sps, off, M, score = _best_symbol_points(sig2, label, default_sps=8)
        print(f"  {label}: expected M={M}, chosen window={chosen}, best sps={sps}, offset={off}, score={score:.3f}, points={len(pts)}")
        if len(pts) > 1200:
            pts = rng.choice(pts, size=1200, replace=False)
        symbol_examples.append((chosen, label, pts, sps, off, M, score))

    cols = min(3, len(symbol_examples))
    rows = int(math.ceil(len(symbol_examples) / cols)) if symbol_examples else 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (_, label, pts, sps, off, M, score) in zip(axes, symbol_examples):
        ax.scatter(np.real(pts), np.imag(pts), s=12, alpha=0.55)
        # Draw ideal PSK reference points after correction.
        ideal = np.exp(1j * 2 * np.pi * np.arange(M) / M)
        ax.scatter(np.real(ideal), np.imag(ideal), s=90, marker="x", linewidths=2)
        ax.set_title(f"Corrected symbols: {label}  M={M}, offset={off}")
        ax.set_xlabel("I / real")
        ax.set_ylabel("Q / imaginary")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.axis("on")
    fig.suptitle("V16 corrected symbol-spaced constellation preview")
    save_or_show(fig, "02_constellation_v16_corrected_symbols", cfg)

    def fft_values(sig):
        win = np.hanning(len(sig))
        spec = np.fft.fftshift(np.fft.fft(sig * win))
        freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1.0))
        mag = 20 * np.log10(np.abs(spec) + 1e-12)
        mag -= np.max(mag)
        return freqs, mag, spec

    def plot_fft(ax, sig, label):
        freqs, mag, _ = fft_values(sig)
        ax.plot(freqs, mag, linewidth=1.0)
        ax.set_title(f"FFT magnitude: {label}")
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Magnitude rel. peak (dB)")
        ax.set_ylim(-90, 5)
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_fft,
        "Separated FFT magnitude by class",
        "03_fft_magnitude_separated",
        cfg,
    )

    def plot_psd(ax, sig, label):
        freqs, _, spec = fft_values(sig)
        psd = (np.abs(spec) ** 2) / max(len(sig), 1)
        psd_db = 10 * np.log10(psd + 1e-12)
        psd_db -= np.max(psd_db)
        ax.plot(freqs, psd_db, linewidth=1.0)
        ax.set_title(f"Spectral density / PSD: {label}")
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("PSD rel. peak (dB)")
        ax.set_ylim(-90, 5)
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_psd,
        "Separated spectral density / PSD by class",
        "04_spectral_density_separated",
        cfg,
    )

    # Spectrogram of first example only, because spectrograms get dense.
    idx0, label0, sig0 = examples[0]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.specgram(np.real(sig0), NFFT=min(256, cfg.window_size // 4), Fs=1.0, noverlap=min(192, cfg.window_size // 8))
    ax.set_title(f"Spectrogram example: {label0}")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Normalized frequency")
    save_or_show(fig, "05_spectrogram_example", cfg)

    def plot_amplitude(ax, sig, label):
        ax.hist(np.abs(sig), bins=60, density=True, histtype="step", linewidth=1.2)
        ax.set_title(f"Amplitude distribution: {label}")
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_amplitude,
        "Separated amplitude distributions by class",
        "06_amplitude_distribution_separated",
        cfg,
    )

    def plot_phase(ax, sig, label):
        ax.hist(np.angle(sig), bins=60, density=True, histtype="step", linewidth=1.2)
        ax.set_title(f"Phase distribution: {label}")
        ax.set_xlabel("Phase (radians)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_phase,
        "Separated phase distributions by class",
        "07_phase_distribution_separated",
        cfg,
    )

    def plot_autocorrelation(ax, sig, label):
        # Normalized autocorrelation magnitude. This helps reveal periodic OFDM/symbol structure.
        seg = sig[: min(len(sig), cfg.window_size)]
        ac = np.correlate(seg, seg, mode="full")
        ac = np.abs(ac[len(ac) // 2:])
        if np.max(ac) > 0:
            ac = ac / np.max(ac)
        max_lag = min(300, len(ac))
        ax.plot(np.arange(max_lag), ac[:max_lag], linewidth=1.0)
        ax.set_title(f"Autocorrelation: {label}")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Normalized correlation")
        ax.grid(True, alpha=0.3)

    _plot_grid_by_class(
        examples,
        plot_autocorrelation,
        "Separated autocorrelation magnitude by class",
        "08_autocorrelation_separated",
        cfg,
    )


def plot_training_results(history: Dict[str, List[float]], y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names: List[str], cfg: Config) -> None:
    print_header("Step 5: Training and model-result plots")

    if history.get("train_loss"):
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, history["train_loss"], marker="o", label="Train loss")
        ax.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
        ax.set_title("Training loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_or_show(fig, "09_training_and_validation_loss", cfg)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(epochs, history["train_acc"], marker="o", label="Train accuracy")
        ax.plot(epochs, history["val_acc"], marker="o", label="Validation accuracy")
        ax.set_title("Training accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        save_or_show(fig, "10_training_and_validation_accuracy", cfg)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_or_show(fig, "11_confusion_matrix", cfg)

    per_class_acc = []
    for cid in range(len(class_names)):
        mask = y_true == cid
        per_class_acc.append(float(np.mean(y_pred[mask] == cid)) if np.any(mask) else 0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, per_class_acc)
    ax.set_title("Per-class validation accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    save_or_show(fig, "12_per_class_accuracy", cfg)

    confidence = probs.max(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidence, bins=30)
    ax.set_title("Prediction confidence distribution")
    ax.set_xlabel("Max softmax probability")
    ax.set_ylabel("Number of validation windows")
    ax.grid(True, alpha=0.3)
    save_or_show(fig, "13_prediction_confidence", cfg)


# =============================================================================
# 7. PREDICTION / INFERENCE
# =============================================================================

def predict_file(file_path: Path, cfg: Config) -> None:
    memory_path = Path(cfg.memory_path)
    if not memory_path.exists():
        raise FileNotFoundError(f"No memory checkpoint found: {memory_path}")
    ckpt = torch.load(memory_path, map_location="cpu")
    class_names = list(ckpt["class_names"])
    model_name = ckpt.get("model_name", cfg.model)
    model = build_model(model_name, len(class_names))
    model.load_state_dict(ckpt["model_state"])
    device = choose_device(cfg.device)
    model.to(device)

    blocks = load_one_file(file_path, cfg)
    if not blocks:
        raise RuntimeError(f"Could not read prediction file: {file_path}")
    X = np.concatenate([b.X for b in blocks], axis=0)
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = DataLoader(IQDataset(X, dummy_y), batch_size=cfg.batch_size, shuffle=False)
    pred, probs = predict_all(model, loader, device)
    counts = {class_names[i]: int((pred == i).sum()) for i in range(len(class_names))}
    mean_probs = probs.mean(axis=0)
    print_header("Prediction results")
    print(f"File: {file_path}")
    print(f"Windows analyzed: {len(X)}")
    print("Predicted window counts:")
    for k, v in counts.items():
        print(f"  {k:20s}: {v}")
    best = int(np.argmax(mean_probs))
    print(f"\nOverall prediction: {class_names[best]}  confidence={mean_probs[best]:.3f}")


# =============================================================================
# 8. SIMPLE UI / ARGUMENTS
# =============================================================================

def ask_for_data_path(default: str) -> str:
    """Small Tkinter picker. Falls back to keyboard input if Tkinter is unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("ML_POF_V16", "Choose your dataset folder or one signal file.")
        folder = filedialog.askdirectory(title="Choose dataset folder")
        if folder:
            root.destroy()
            return folder
        file = filedialog.askopenfilename(title="Choose one data file")
        root.destroy()
        return file or default
    except Exception:
        typed = input(f"Enter data folder/file path [{default}]: ").strip()
        return typed or default


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    parser = argparse.ArgumentParser(description="Universal POF / OFDM / SDR signal classifier")
    parser.add_argument("--data", default=None, help="Dataset folder or file path")
    parser.add_argument("--memory-path", default=str(DEFAULT_MODEL_DIR / DEFAULT_MEMORY_NAME), help="Checkpoint path, default memory.pt")
    parser.add_argument("--plots-dir", default=str(DEFAULT_PLOTS_DIR), help="Folder for saved plots")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-windows-per-file", type=str, default="200", help="Number or None")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["trnn", "cnn"], default="trnn")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots to --plots-dir. Default is display-only, no graph files saved.")
    parser.add_argument("--no-save-plots", action="store_true", help="Compatibility option; V13 already does not save plots by default.")
    parser.add_argument("--no-show-plots", action="store_true")
    parser.add_argument("--no-train", action="store_true", help="Load and inspect data but do not train")
    parser.add_argument("--predict", default=None, help="Run prediction on a file using memory.pt")
    parser.add_argument("--allow-unknown-labels", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    max_windows: Optional[int]
    if str(args.max_windows_per_file).lower() in {"none", "all", "0", "-1"}:
        max_windows = None
    else:
        max_windows = int(args.max_windows_per_file)

    data = args.data if args.data else ask_for_data_path(str(DEFAULT_DATA_DIR))
    return Config(
        data=data,
        memory_path=args.memory_path,
        plots_dir=args.plots_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_windows_per_file=max_windows,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        seed=args.seed,
        model=args.model,
        save_plots=bool(args.save_plots and not args.no_save_plots),
        show_plots=not args.no_show_plots,
        no_train=args.no_train,
        prediction_file=args.predict,
        allow_unknown_labels=args.allow_unknown_labels,
        device=args.device,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = parse_args(argv)
    set_seed(cfg.seed)
    print_header("ML_POF_V16 Universal POF / SDR Signal Classifier")
    print(json.dumps(asdict(cfg), indent=2))

    if cfg.prediction_file:
        predict_file(Path(cfg.prediction_file), cfg)
        return

    X, y, snr, class_names, meta = build_dataset(cfg)
    plot_data_preview(X, y, snr, class_names, cfg)

    if cfg.no_train:
        print("\n--no-train was selected. Data loading and preview are complete.")
        return

    if len(np.unique(y)) < 2:
        raise RuntimeError("Training needs at least 2 classes. Add labeled data for another class or use --no-train for inspection.")

    model, history, val_results = train_model(X, y, class_names, cfg, meta)
    y_true, y_pred, probs = val_results
    plot_training_results(history, y_true, y_pred, probs, class_names, cfg)

    print_header("Done")
    print(f"Saved model memory: {Path(cfg.memory_path)}")
    if cfg.save_plots:
        print(f"Saved plots folder: {Path(cfg.plots_dir)}")
    print("Recommended next step: increase --max-windows-per-file and --epochs after the first successful small test.")


if __name__ == "__main__":
    main()
