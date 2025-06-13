"""Forecast-error metrics (RMSE, MAE, etc.).

Verbatim copy of the legacy *src/metrics_utils.py* so other modules can
``from iv_drl.utils.metrics_utils import rmse``.
"""

from __future__ import annotations

import numpy as np

__all__ = ["rmse", "mae", "rmae"]


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root-mean-squared error."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute error."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(y - yhat)))


def rmae(y: np.ndarray, yhat: np.ndarray) -> float:
    """Root mean absolute error (sqrt(MAE))."""
    return float(np.sqrt(mae(y, yhat))) 