# Rewritten LSTM benchmark with multivariate inputs, longer sequence window,
# validation early-stopping and proper scaling.
from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from iv_drl.utils import load_config

CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()

# ---------------------------------------------------------------------
# Hyper-parameters (can be tweaked via CLI kwargs later if needed)
# ---------------------------------------------------------------------
SEQ_LEN = 60
BATCH_SIZE = 64
MAX_EPOCHS = 200
LR = 1e-3
PATIENCE = 10  # early-stopping on *validation* loss
HIDDEN = 128
NUM_LAYERS = 2
DROPOUT = 0.2
VAL_RATIO = 0.15  # of *train* window; test/OOS is remaining 1-train_ratio
TRAIN_RATIO = 0.7  # chronological split between train+val and OOS

# Global random seed (overridden by --seed CLI)
SEED = 42

def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for full reproducibility."""

    global SEED  # noqa: PLW0603
    SEED = int(seed)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

# Call once so module-level ops are deterministic; will be *overwritten* if
# the user passes a different --seed later.
_set_seed(SEED)

# ------------------------------------------------------------------
# GPU / CPU device
# ------------------------------------------------------------------
DEVICE = torch.device("cpu")
print("Using", DEVICE)

def tensor(arr):
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Return 3-D array (N, seq_len, F) and 1-D targets (N,)"""
    seq_X, seq_y = [], []
    for i in range(seq_len, len(y)):
        seq_X.append(X[i - seq_len : i])
        seq_y.append(y[i])
    return np.asarray(seq_X, dtype=np.float32), np.asarray(seq_y, dtype=np.float32)


class _LSTMNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(HIDDEN)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])  # last time-step
        return self.fc(out).squeeze(-1)  # (B,)


def _standardize(train_arr: np.ndarray, *other: np.ndarray):
    """Z-score using mean/std from *train*.

    Returns the scaled arrays (train, *other) and the fitted (mean, std).
    """
    mean = train_arr.mean(axis=(0, 1), keepdims=True)
    std = train_arr.std(axis=(0, 1), keepdims=True) + 1e-8
    rescaled = [(arr - mean) / std for arr in (train_arr, *other)]
    return rescaled, (mean.squeeze(), std.squeeze())


def run_lstm(*, out_csv: str | Path | None = None, param_file: str | None = None, seed: int | None = None) -> Path:
    """Train the enhanced LSTM baseline.

    If ``param_file`` is provided (JSON), hyper-parameters in the file will
    override the global defaults defined at the top of this script.  Expected
    keys: seq_len, hidden, num_layers, dropout, lr, batch_size, weight_decay.
    """

    global SEQ_LEN, HIDDEN, NUM_LAYERS, DROPOUT, LR, BATCH_SIZE  # noqa: PLW0603

    if param_file:
        try:
            with open(param_file, "r", encoding="utf-8") as fh:
                params = json.load(fh)
            SEQ_LEN = int(params.get("seq_len", SEQ_LEN))
            HIDDEN = int(params.get("hidden", HIDDEN))
            NUM_LAYERS = int(params.get("num_layers", NUM_LAYERS))
            DROPOUT = float(params.get("dropout", DROPOUT))
            LR = float(params.get("lr", LR))
            BATCH_SIZE = int(params.get("batch_size", BATCH_SIZE))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load param file {param_file}: {exc}")

        # Debug: show final hyper-parameters in effect
        print(
            f"[DEBUG] LSTM params → seq_len={SEQ_LEN}, hidden={HIDDEN}, num_layers={NUM_LAYERS}, "
            f"dropout={DROPOUT}, lr={LR}, batch_size={BATCH_SIZE}"
        )

    # If called as a library function, allow overriding the RNG seed here.
    if seed is not None:
        _set_seed(seed)

    # ------------------------------------------------------------------
    # 1. Load data and assemble feature matrix identical to DRL env state
    # ------------------------------------------------------------------
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")

    feature_cols = CFG["features"]["all_feature_cols"]
    # ensure all columns present
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    print(f"Available features: {len(available_cols)}/{len(feature_cols)}")
    print(f"Missing features: {missing_cols}")
    feature_cols = available_cols
    
    # Handle remaining NaNs in macro features
    macro_cols = ['vol_of_vol', 'vix_ts_slope']
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    X_full = df[feature_cols].astype(float).to_numpy()

    # target is next-day ATM-IV
    target_col = CFG["features"]["target_col"]
    y_full = df[target_col].shift(-1).astype(float).to_numpy()
    dates_full = df["date"].shift(-1).to_numpy()

    # remove last NaN row (no next-day target)
    mask = ~np.isnan(y_full)
    X_full, y_full, dates_full = X_full[mask], y_full[mask], dates_full[mask]

    # ------------------------------------------------------------------
    # 2. Build sequences of length SEQ_LEN
    # ------------------------------------------------------------------
    X_seq, y_seq = _build_sequences(X_full, y_full, SEQ_LEN)
    dates_seq = dates_full[SEQ_LEN:]

    # ------------------------------------------------------------------
    # 3. Chronological split → train/val/OOS
    # ------------------------------------------------------------------
    n_total = len(X_seq)
    train_end = int(TRAIN_RATIO * n_total)
    oos_X, oos_y, oos_dates = X_seq[train_end:], y_seq[train_end:], dates_seq[train_end:]

    train_X_full, train_y_full = X_seq[:train_end], y_seq[:train_end]
    # further split train into train/val
    val_split = int((1 - VAL_RATIO) * len(train_X_full))
    tr_X, val_X = train_X_full[:val_split], train_X_full[val_split:]
    tr_y, val_y = train_y_full[:val_split], train_y_full[val_split:]

    # ------------------------------------------------------------------
    # 4. Standardise using only *training* data statistics
    # ------------------------------------------------------------------
    (tr_X, val_X, oos_X), _ = _standardize(tr_X, val_X, oos_X)

    # convert to torch tensors
    tr_ds = TensorDataset(tensor(tr_X), tensor(tr_y))
    val_ds = TensorDataset(tensor(val_X), tensor(val_y))
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = _LSTMNet(input_dim=tr_X.shape[-1]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss()

    best_state = model.state_dict()  # Initialize with current state
    best_val = float("inf")
    patience = PATIENCE

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()

        # ----- validation -----
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_losses.append(loss_fn(model(xb), yb).item())
            val_loss = float(np.mean(v_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:  # small tolerance
            best_val = val_loss
            best_state = model.state_dict()
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    if best_state is not None:  # Only load if we have a best state
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # 5. Forecast on OOS sequences
    # ------------------------------------------------------------------
    print(f"oos_X shape: {oos_X.shape}")
    print(f"Any NaN in oos_X: {np.isnan(oos_X).any()}")
    print(f"Any Inf in oos_X: {np.isinf(oos_X).any()}")
    # Print which features have NaNs in oos_X
    nan_features = []
    for i, col in enumerate(feature_cols):
        if np.isnan(oos_X[:, :, i]).any():
            nan_features.append(col)
    print(f"Features with NaNs in oos_X: {nan_features}")
    model.eval()
    with torch.no_grad():
        preds = model(tensor(oos_X)).cpu().numpy()
        print(f"Prediction shape: {preds.shape}")
        print(f"First few predictions: {preds[:5]}")
        print(f"Any NaN predictions: {np.isnan(preds).any()}")

    out_path = Path(out_csv) if out_csv else OUT_DIR / "lstm_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": oos_dates, "lstm_forecast": preds}).to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print('Saved LSTM forecasts to', out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--param_file", type=str, default=None, help="JSON file of tuned hyper-parameters")
    p.add_argument("--out_csv", type=str, default=None, help="Override output csv path")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed for full reproducibility")
    args = p.parse_args()
    _set_seed(args.seed)
    run_lstm(out_csv=args.out_csv, param_file=args.param_file, seed=args.seed) 