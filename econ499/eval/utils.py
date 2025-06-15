from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from econ499.utils import load_config

CONFIG = load_config("data_config.yaml")
DATA_DIR = Path(CONFIG["paths"]["output_dir"]).resolve()


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def _find_forecast_col(df: pd.DataFrame) -> str | None:
    others = [c for c in df.columns if c.lower() != "date"]
    if len(others) == 1:
        return others[0]
    for c in others:
        if c.endswith("_forecast"):
            return c
    return None


def _load_predictions() -> list[tuple[str, pd.Series]]:
    out = []
    files = list(DATA_DIR.glob("*_oos_predictions.csv"))
    for csv in files:
        try:
            df = pd.read_csv(csv, parse_dates=["date"])
        except Exception:
            continue
        if df.empty or "date" not in df.columns:
            continue
        col = _find_forecast_col(df)
        if col is None:
            continue
        name = col.replace("_forecast", "")
        out.append((name, df[["date", col]].rename(columns={col: name})))
    return out


def _qlike(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float:
    """QLIKE loss (Patton & Sheppard, 2009) for volatility forecasts."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float).clip(min=eps)
    return float(np.mean(np.log(yhat) + y / yhat))
