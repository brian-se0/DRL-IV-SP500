from __future__ import annotations

import numpy as np
from scipy.stats import t
from statsmodels.tsa.stattools import acovf

__all__ = ["dm_test", "spa_test", "mcs_loss_set"]


def _newey_west_variance(d: np.ndarray, lag: int) -> float:
    """Newey–West HAC variance estimator for the loss-diff series *d*.

    Parameters
    ----------
    d : np.ndarray
        Loss differential series.
    lag : int
        Truncation lag (usually forecast horizon h – 1).
    """
    d = d - np.mean(d)
    n = len(d)
    gamma = acovf(d, nlag=lag, fft=False)
    var = gamma[0] + 2 * np.sum((1 - np.arange(1, lag + 1) / (lag + 1)) * gamma[1:])
    return var


def dm_test(
    y: np.ndarray,
    yhat1: np.ndarray,
    yhat2: np.ndarray,
    *,
    h: int = 1,
    power: int = 2,
) -> tuple[float, float]:
    """Small-sample corrected Diebold–Mariano test.

    Parameters
    ----------
    y : array-like of shape (n,)
        True values.
    yhat1, yhat2 : array-like
        Competing forecasts.
    h : int, default 1
        Forecast horizon.
    power : {1, 2}, default 2
        Loss function exponent; 2 ⇒ squared error, 1 ⇒ absolute error.

    Returns
    -------
    t_stat : float
        DM t-statistic (should follow t-distribution with df = n-1).
    p_value : float
        Two-sided p-value.
    """
    y = np.asarray(y, dtype=float)
    yhat1 = np.asarray(yhat1, dtype=float)
    yhat2 = np.asarray(yhat2, dtype=float)

    if power == 2:
        loss = lambda a, b: (a - b) ** 2  # noqa: E731
    elif power == 1:
        loss = lambda a, b: np.abs(a - b)  # noqa: E731
    else:
        raise ValueError("power must be 1 or 2")

    d = loss(y, yhat1) - loss(y, yhat2)
    n = len(d)
    mean_d = np.mean(d)

    var_d = _newey_west_variance(d, lag=h - 1)
    dm_stat = mean_d / np.sqrt(var_d / n)

    # Harvey, Leybourne & Newbold small-sample adjustment
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t_stat = dm_stat * adj
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=n - 1))

    return float(t_stat), float(p_val)


# -----------------------------------------------------------------------------
# Hansen (2005) Superior Predictive Ability (SPA) test – naive bootstrap version
# -----------------------------------------------------------------------------

def spa_test(
    y: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    benchmark: str,
    loss: str = "mse",
    B: int = 500,
    seed: int | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    model_names = [k for k in forecasts if k != benchmark]
    
    if loss == "mse":
        loss_fn = lambda a, b: (a - b) ** 2  # noqa: E731
    elif loss == "mae":
        loss_fn = lambda a, b: np.abs(a - b)  # noqa: E731
    else:
        raise ValueError("loss must be 'mse' or 'mae'")
    
    L_bench = loss_fn(y, forecasts[benchmark])
    d = np.vstack([
        loss_fn(y, forecasts[m]) - L_bench for m in model_names
    ])  # shape (M, T)
    Tn = d.shape[1]
    t0 = d.mean(axis=1) / (d.std(axis=1, ddof=1) / np.sqrt(Tn))
    
    t_boot = np.empty((len(model_names), B))
    for b in range(B):
        idx = rng.integers(0, Tn, size=Tn)
        d_b = d[:, idx]
        t_boot[:, b] = d_b.mean(axis=1) / (d_b.std(axis=1, ddof=1) / np.sqrt(Tn))
    return {m: float(np.mean(t_boot[i] > t0[i])) for i, m in enumerate(model_names)}


# -----------------------------------------------------------------------------
# Model Confidence Set (MCS) – simple bootstrap TR statistic
# -----------------------------------------------------------------------------

def mcs_loss_set(
    y: np.ndarray,
    forecasts: dict[str, np.ndarray],
    *,
    alpha: float = 0.10,
    B: int = 500,
    seed: int | None = None,
) -> list[str]:
    rng = np.random.default_rng(seed)
    models = list(forecasts.keys())
    losses = {m: (y - forecasts[m]) ** 2 for m in models}
    Tn = len(y)

    def t_stat(arr: np.ndarray) -> float:
        return np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(Tn))

    while len(models) > 1:
        avg_loss = np.mean([losses[m] for m in models], axis=0)
        d = {m: losses[m] - avg_loss for m in models}
        tr_stat = max(abs(t_stat(d[m])) for m in models)
        # bootstrap distribution
        tr_boot = np.empty(B)
        for b in range(B):
            idx = rng.integers(0, Tn, size=Tn)
            tr_boot[b] = max(abs(t_stat(d[m][idx])) for m in models)
        p_val = np.mean(tr_boot > tr_stat)
        if p_val < alpha:
            # remove worst performer (highest mean SE)
            worst = max(models, key=lambda m: np.mean(losses[m]))
            models.remove(worst)
        else:
            break
    return models
