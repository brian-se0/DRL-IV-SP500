from __future__ import annotations

"""Hyper-parameter optimisation for the FFN baseline using Optuna.

Run:
    python -m econ499.hpo.hpo_ffn --trials 50
This creates an SQLite study in ``results/`` and writes the best params to
``results/best_ffn_params.json``.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from econ499.utils import load_config

CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUTPUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STUDY_NAME = "ffn_iv_hpo"
STUDY_DB = OUTPUT_DIR / f"{STUDY_NAME}.db"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_ffn_params.json"

DEVICE = torch.device("cpu")


def tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def _standardize(train_arr: np.ndarray, *others: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-8
    rescaled = [(arr - mean) / std for arr in (train_arr, *others)]
    return rescaled, (mean.squeeze(), std.squeeze())


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_data() -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    feat_cols = [c for c in CFG["features"]["all_feature_cols"] if c in df.columns]
    X_full = df[feat_cols].astype(float).to_numpy()
    target_col = CFG["features"]["target_col"]
    y_full = df[target_col].shift(-1).astype(float).to_numpy()
    mask = ~np.isnan(y_full)
    return X_full[mask], y_full[mask]


def _objective(trial: optuna.Trial) -> float:
    hidden = trial.suggest_categorical("hidden", [32, 64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    max_epochs = trial.suggest_int("max_epochs", 50, 200)

    X_full, y_full = _load_data()
    n_total = len(X_full)
    train_end = int(0.7 * n_total)
    train_X_full = X_full[:train_end]
    train_y_full = y_full[:train_end]

    val_split = int((1 - 0.15) * len(train_X_full))
    tr_X, val_X = train_X_full[:val_split], train_X_full[val_split:]
    tr_y, val_y = train_y_full[:val_split], train_y_full[val_split:]

    (tr_X, val_X), _ = _standardize(tr_X, val_X)

    train_loader = DataLoader(TensorDataset(tensor(tr_X), tensor(tr_y)), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(tensor(val_X), tensor(val_y)), batch_size=batch_size, shuffle=False)

    model = FFN(tr_X.shape[1], hidden, dropout).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    patience = 8
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_losses = [loss_fn(model(xb), yb).item() for xb, yb in val_loader]
        val_mae = float(np.mean(val_losses))
        trial.report(val_mae, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_mae < best_val - 1e-4:
            best_val = val_mae
            patience = 8
        else:
            patience -= 1
            if patience == 0:
                break

    return best_val


def main(trials: int) -> None:
    study = optuna.create_study(direction="minimize", study_name=STUDY_NAME, storage=f"sqlite:///{STUDY_DB}", load_if_exists=True)
    study.optimize(_objective, n_trials=trials, show_progress_bar=True)
    print("Best value", study.best_value)
    print("Best params", study.best_params)
    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print("Best params saved to", BEST_PARAMS_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    main(args.trials)
