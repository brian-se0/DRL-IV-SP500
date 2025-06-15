from __future__ import annotations

"""Simple feed-forward neural network baseline for ATM-IV forecasting."""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from econ499.utils import load_config

CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()

# ----------------------------------------------------------------------
# default hyper-parameters
# ----------------------------------------------------------------------
BATCH_SIZE = 64
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 10
VAL_RATIO = 0.15
TRAIN_RATIO = 0.7

SEED = 42


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""

    global SEED  # noqa: PLW0603
    SEED = int(seed)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


_set_seed(SEED)

DEVICE = torch.device("cpu")
print("Using", DEVICE)


def tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


class _FFN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, F)
        return self.net(x).squeeze(-1)  # (B,)


# ----------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------

def _standardize(train_arr: np.ndarray, *others: np.ndarray):
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True) + 1e-8
    rescaled = [(arr - mean) / std for arr in (train_arr, *others)]
    return rescaled, (mean.squeeze(), std.squeeze())


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------

def run_ffn(
    *,
    out_csv: str | Path | None = None,
    param_file: str | None = None,
    seed: int | None = None,
    max_epochs: int | None = None,
) -> Path:
    """Train FFN on historical panel and save OOS forecasts."""

    global BATCH_SIZE, HIDDEN, DROPOUT, LR  # noqa: PLW0603

    if param_file:
        try:
            with open(param_file, "r", encoding="utf-8") as fh:
                params = json.load(fh)
            BATCH_SIZE = int(params.get("batch_size", BATCH_SIZE))
            HIDDEN = int(params.get("hidden", HIDDEN))
            DROPOUT = float(params.get("dropout", DROPOUT))
            LR = float(params.get("lr", LR))
            if "max_epochs" in params:
                max_epochs = int(params["max_epochs"])
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not load param file {param_file}: {exc}")
        print(
            f"[DEBUG] FFN params → hidden={HIDDEN}, dropout={DROPOUT}, lr={LR}, batch_size={BATCH_SIZE}"
        )

    if seed is not None:
        _set_seed(seed)

    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")

    feature_cols = [c for c in CFG["features"]["all_feature_cols"] if c in df.columns]
    X_full = df[feature_cols].astype(float).to_numpy()
    target_col = CFG["features"]["target_col"]
    y_full = df[target_col].shift(-1).astype(float).to_numpy()
    dates_full = df["date"].shift(-1).to_numpy()

    mask = ~np.isnan(y_full)
    X_full, y_full, dates_full = X_full[mask], y_full[mask], dates_full[mask]

    n_total = len(X_full)
    train_end = int(TRAIN_RATIO * n_total)
    oos_X = X_full[train_end:]
    oos_dates = dates_full[train_end:]
    train_X_full = X_full[:train_end]
    train_y_full = y_full[:train_end]

    val_split = int((1 - VAL_RATIO) * len(train_X_full))
    tr_X, val_X = train_X_full[:val_split], train_X_full[val_split:]
    tr_y, val_y = train_y_full[:val_split], train_y_full[val_split:]

    (tr_X, val_X, oos_X), _ = _standardize(tr_X, val_X, oos_X)

    tr_ds = TensorDataset(tensor(tr_X), tensor(tr_y))
    val_ds = TensorDataset(tensor(val_X), tensor(val_y))
    train_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = _FFN(input_dim=tr_X.shape[1]).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)
    loss_fn = nn.SmoothL1Loss()

    best_state = None
    best_val = float("inf")
    patience = PATIENCE

    epochs = max_epochs or MAX_EPOCHS
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_losses = [loss_fn(model(xb), yb).item() for xb, yb in val_loader]
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = model.state_dict()
            patience = PATIENCE
        else:
            patience -= 1
            if patience == 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(tensor(oos_X)).cpu().numpy()

    out_path = Path(out_csv) if out_csv else OUT_DIR / "ffn_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": oos_dates, "ffn_forecast": preds}).to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print('Saved FFN forecasts to', out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--param_file", type=str, default=None, help="JSON file of tuned hyper-parameters")
    p.add_argument("--out_csv", type=str, default=None, help="Override output csv path")
    p.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    p.add_argument("--max_epochs", type=int, default=None, help="Override training epochs")
    args = p.parse_args()
    _set_seed(args.seed)
    run_ffn(out_csv=args.out_csv, param_file=args.param_file, seed=args.seed, max_epochs=args.max_epochs)
