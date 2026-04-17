#!/usr/bin/env python3
"""
Train/evaluate IDS models (DNN, LSTM, Hybrid).

Examples:
  python train.py --model all
  python train.py --model hybrid --csv /path/to/KDDTrain+.csv --label label

Without --csv, uses synthetic sequences for a quick smoke test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_csv_supervised, make_synthetic_sequences
from src.metrics import binary_metrics
from src.models import DNNClassifier, HybridDNNLSTM, LSTMClassifier

import time


def benchmark_forward_ms(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    warmup: int = 20,
    iters: int = 100,
    is_sequence: bool,
) -> float:
    """Mean wall-clock ms per sample for forward-only (eval mode)."""
    model.eval()
    n = len(X)
    x = torch.from_numpy(X[: min(n, batch_size * 4)]).float().to(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x)
        t1 = time.perf_counter()
    total_samples = iters * x.shape[0]
    return (t1 - t0) * 1000.0 / total_samples


def train_one(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    is_sequence: bool,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray]:
    model = model.to(device)
    if is_sequence:
        xt = torch.from_numpy(X_train).float()
        xv = torch.from_numpy(X_val).float()
    else:
        xt = torch.from_numpy(X_train).float()
        xv = torch.from_numpy(X_val).float()
    yt = torch.from_numpy(y_train).long()
    yv = torch.from_numpy(y_val).long()

    # class weights for imbalance
    n0 = (y_train == 0).sum()
    n1 = (y_train == 1).sum()
    w0 = len(y_train) / (2 * max(n0, 1))
    w1 = len(y_train) / (2 * max(n1, 1))
    weight = torch.tensor([w0, w1], device=device, dtype=torch.float32)

    loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(xv.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return model, pred


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dnn", "lstm", "hybrid", "all"], default="all")
    p.add_argument("--csv", type=str, default="", help="Path to CSV (NSL-KDD / CICIDS preprocessed)")
    p.add_argument("--label", type=str, default="label", help="Label column name in CSV")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--bench",
        action="store_true",
        help="After training, print mean forward-pass ms/sample for each model (GPU/CPU dependent).",
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.csv:
        X_tr_dnn, X_te_dnn, y_tr, y_te = load_csv_supervised(args.csv, label_column=args.label)
        n_features = X_tr_dnn.shape[1]
        seq_len = 1
        X_tr_seq = X_tr_dnn.reshape(len(X_tr_dnn), seq_len, n_features)
        X_te_seq = X_te_dnn.reshape(len(X_te_dnn), seq_len, n_features)
        dnn_in = n_features
    else:
        X_tr_dnn, X_te_dnn, X_tr_seq, X_te_seq, y_tr, y_te = make_synthetic_sequences()
        _, _, n_features = X_tr_seq.shape
        dnn_in = X_tr_dnn.shape[1]

    models_to_run: list[tuple[str, nn.Module, np.ndarray, np.ndarray, bool]] = []
    if args.model in ("dnn", "all"):
        models_to_run.append(("DNN", DNNClassifier(dnn_in), X_tr_dnn, X_te_dnn, False))
    if args.model in ("lstm", "all"):
        models_to_run.append(("LSTM", LSTMClassifier(n_features), X_tr_seq, X_te_seq, True))
    if args.model in ("hybrid", "all"):
        models_to_run.append(("Hybrid", HybridDNNLSTM(n_features), X_tr_seq, X_te_seq, True))

    for name, model, Xtr, Xte, is_seq in models_to_run:
        _, pred = train_one(
            model,
            Xtr,
            y_tr,
            Xte,
            y_te,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            is_sequence=is_seq,
            device=device,
        )
        m = binary_metrics(y_te, pred)
        print(
            f"{name:6s}  acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}  fpr={m['fpr']:.4f}"
        )
        if args.bench:
            ms = benchmark_forward_ms(model, Xte, device, is_sequence=is_seq)
            print(f"        forward: {ms:.4f} ms/sample (batch={min(256, len(Xte))}, see --bench)")


if __name__ == "__main__":
    main()
