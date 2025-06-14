from __future__ import annotations

"""Hyper-parameter optimisation for the LSTM baseline using Optuna.

Run:
    python -m iv_drl.tune.hpo_lstm --trials 50
This creates an SQLite study in ``data_processed/`` (same folder as other
HPO studies) and writes the best params to
``data_processed/best_lstm_params.json``.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import pandas as pd

from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import mae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUTPUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_NAME = "lstm_iv_hpo"
STUDY_DB = OUTPUT_DIR / f"{STUDY_NAME}.db"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_lstm_params.json"

torch.manual_seed(42)
np.random.seed(42)

# -----------------------------------------------------------------------------
# device
# -----------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # force CPU to avoid GPU issues
print("Using device:", DEVICE)

# -----------------------------------------------------------------------------
# data helpers
# -----------------------------------------------------------------------------

def build_sequences(X: np.ndarray, y: np.ndarray, seq: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(seq, len(y)):
        xs.append(X[i - seq : i])
        ys.append(y[i])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def zscore(train: np.ndarray, *others: np.ndarray):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-8
    return [(arr - mean) / std for arr in (train, *others)]


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------

class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # x (B,T,F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1]).squeeze(-1)


# -----------------------------------------------------------------------------
# objective
# -----------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    # hyper-params
    seq_len = trial.suggest_categorical("seq_len", [60, 90, 120])
    hidden = trial.suggest_categorical("hidden", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # data load
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    feat_cols = CFG["features"]["all_feature_cols"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    
    # Handle missing values in features
    # For price-based features, use forward fill
    price_cols = ['Close', 'Volume', 'log_return', 'garman_klass_rv', 'parkinson_rv', 'amihud_illiq']
    price_cols = [c for c in price_cols if c in feat_cols]
    if price_cols:
        df[price_cols] = df[price_cols].ffill()
    
    # For derived features, don't fill - they should be calculated correctly
    derived_cols = ['VIX_daily_change', 'vol_of_vol', 'term_spread_10y_3m', 'credit_spread_baa_aaa']
    derived_cols = [c for c in derived_cols if c in feat_cols]
    
    # For other features, use forward fill
    other_cols = [c for c in feat_cols if c not in price_cols and c not in derived_cols]
    if other_cols:
        df[other_cols] = df[other_cols].ffill()
    
    # Convert to numpy arrays
    X_full = df[feat_cols].astype(float).to_numpy()
    target_col = CFG["features"]["target_col"]
    y_full = df[target_col].shift(-1).astype(float).to_numpy()
    
    # Remove any remaining NaN values
    mask = ~np.isnan(y_full) & ~np.isnan(X_full).any(axis=1)
    X_full, y_full = X_full[mask], y_full[mask]
    
    if len(X_full) == 0:
        raise ValueError("No valid data after preprocessing")

    X_seq, y_seq = build_sequences(X_full, y_full, seq_len)

    # chrono split: 70 % train, 15 % val, 15 % test (test unused here)
    n_total = len(X_seq)
    train_end = int(0.7 * n_total)
    val_end = int(0.85 * n_total)
    tr_X, tr_y = X_seq[:train_end], y_seq[:train_end]
    val_X, val_y = X_seq[train_end:val_end], y_seq[train_end:val_end]

    # scale
    tr_X, val_X = zscore(tr_X, val_X)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    train_loader = DataLoader(TensorDataset(tensor(tr_X), tensor(tr_y)), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(tensor(val_X), tensor(val_y)), batch_size=batch_size, shuffle=False)

    model = LSTMNet(tr_X.shape[-1], hidden, num_layers, dropout).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    best_val = math.inf
    patience, max_patience = 8, 8
    for epoch in range(150):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
        # validation
        model.eval()
        with torch.no_grad():
            v_losses = []
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                v_losses.append(loss_fn(model(xb), yb).item())
        v_mae = float(np.mean(v_losses))
        trial.report(v_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if v_mae < best_val - 1e-4:
            best_val = v_mae
            patience = max_patience
        else:
            patience -= 1
            if patience == 0:
                break
    return best_val


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(trials: int):
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}", load_if_exists=True)
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    print("Best value", study.best_value)
    print("Best params", study.best_params)
    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print('Best params saved to', BEST_PARAMS_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()
    main(args.trials) 