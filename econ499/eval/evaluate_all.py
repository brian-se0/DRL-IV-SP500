"""Aggregate forecasts and compute error metrics.

Replaces old *src/evaluate_forecasts.py*.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from econ499.utils import load_config
from econ499.utils.metrics_utils import rmse, mae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(CONFIG["paths"]["output_dir"]).resolve()
ARTIFACT_DIR = ROOT / "artifacts" / "tables"
TARGET_COL = CONFIG["features"]["target_col"]


def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def _find_forecast_col(df: pd.DataFrame) -> str:
    others = [c for c in df.columns if c.lower() != "date"]
    if len(others) == 1:
        return others[0]
    for c in others:
        if c.endswith("_forecast"):
            return c
    raise ValueError("Unable to identify forecast column")


def _load_predictions() -> list[tuple[str, pd.Series]]:
    out = []
    for csv in DATA_DIR.glob("*_oos_predictions.csv"):
        try:
            df = pd.read_csv(csv, parse_dates=["date"])
        except Exception:
            continue
        if df.empty or "date" not in df.columns:
            continue
        col = _find_forecast_col(df)
        name = col.replace("_forecast", "")
        out.append((name, df[["date", col]].rename(columns={col: name})))
    return out


def _qlike(y: np.ndarray, yhat: np.ndarray, eps: float = 1e-12) -> float:
    """QLIKE loss (Patton & Sheppard, 2009) for volatility forecasts.

    Parameters
    ----------
    y : np.ndarray
        Realised volatility proxy (here next-day ATM-IV).
    yhat : np.ndarray
        Forecasts (same scale as *y*).
    eps : float, optional
        Floor to avoid log/zero division when predictions are ~0.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float).clip(min=eps)
    return float(np.mean(np.log(yhat) + y / yhat))


def evaluate_all(panel_csv: str | Path | None = None, **kwargs) -> Path:
    """Compute error metrics for every *_oos_predictions.csv* in data_processed.

    Returns
    -------
    Path of the saved metrics CSV.
    """
    if panel_csv:
        panel_path = Path(panel_csv)
    else:
        default_path = Path(CONFIG["paths"]["drl_state_file"]).resolve()
        alt_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "spx_iv_drl_state.csv"
        panel_path = default_path if default_path.exists() else alt_path
    panel = pd.read_csv(panel_path, parse_dates=["date"]).sort_values("date")
    panel["IV_next"] = panel[TARGET_COL].shift(-1)
    panel.dropna(subset=["IV_next"], inplace=True)

    eval_df = panel[["date", "IV_next"]].copy()

    # naive (today's IV)
    eval_df["naive"] = panel[TARGET_COL].values
    # AR(1) rolling mean of last two obs
    eval_df["ar1"] = (
        panel[TARGET_COL].shift(1).rolling(2, min_periods=1).mean().bfill()
    )

    for name, df_pred in _load_predictions():
        eval_df = eval_df.merge(df_pred, on="date", how="left")
    eval_df.dropna(inplace=True)

    y_true = eval_df["IV_next"].to_numpy()

    naive_mae = mae(y_true, eval_df["naive"].to_numpy())

    rows = []
    for col in eval_df.columns:
        if col in {"date", "IV_next"}:
            continue
        yhat = eval_df[col].to_numpy()
        mae_model = mae(y_true, yhat)
        mase = mae_model / naive_mae
        rows.append([
            col,
            rmse(y_true, yhat),
            mae_model,
            mase,
            _mape(y_true, yhat),
            _qlike(y_true, yhat),
        ])

    metrics = (
        pd.DataFrame(
            rows,
            columns=["model", "RMSE", "MAE", "MASE", "MAPE(%)", "QLIKE"],
        )
        .set_index("model")
        .sort_values("RMSE")
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "forecast_metrics.csv"
    metrics.to_csv(out_path, float_format="%.6f")

    print(metrics.to_markdown(floatfmt=".6f"))
    print("Metrics saved to", out_path)

    # ---------------- Diebold–Mariano (optional) ----------------
    if kwargs.get("dm_base") is not None:
        from econ499.eval.stat_tests import dm_test  # local helper

        base = kwargs["dm_base"]
        if base not in eval_df.columns:
            raise ValueError(f"dm_base '{base}' not found in evaluation columns")

        print("\n### Diebold–Mariano tests vs", base)
        for col in eval_df.columns:
            if col in {"date", "IV_next", base}:
                continue
            t_stat, p_val = dm_test(
                y_true,
                eval_df[base].to_numpy(),
                eval_df[col].to_numpy(),
                h=kwargs.get("dm_lag", 1),
            )
            print(f"{col:10s}  t={t_stat:+.3f}  p={p_val:.4f}")

    # ---------------- SPA / MCS (optional) ----------------
    if kwargs.get("spa_base") is not None:
        from econ499.eval.stat_tests import spa_test

        bench = kwargs["spa_base"]
        forecasts_dict = {c: eval_df[c].to_numpy() for c in eval_df.columns if c not in {"date", "IV_next"}}
        pvals = spa_test(y_true, forecasts_dict, benchmark=bench, B=kwargs.get("spa_B", 500))
        print("\n### SPA test p-values (benchmark:", bench, ")")
        for m, pv in pvals.items():
            print(f"{m:10s}  p={pv:.4f}")

    if kwargs.get("mcs", False):
        from econ499.eval.stat_tests import mcs_loss_set
        forecasts_dict = {c: eval_df[c].to_numpy() for c in eval_df.columns if c not in {"date", "IV_next"}}
        mcs_set = mcs_loss_set(y_true, forecasts_dict, alpha=kwargs.get("mcs_alpha", 0.10))
        print(f"\n### Model Confidence Set (alpha={kwargs.get('mcs_alpha', 0.10):.2f}) => {mcs_set}")

        # Flag retained models so downstream figure scripts can read directly
        metrics["mcs_in_set"] = 0
        for m in mcs_set:
            if m in metrics.index:
                metrics.at[m, "mcs_in_set"] = 1
        # Overwrite CSV with the new column
        metrics.to_csv(out_path, float_format="%.6f")
        print("[INFO] Added 'mcs_in_set' column to", out_path)

    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv", type=str, default=None, help="Override merged panel path")
    p.add_argument("--dm_base", type=str, default=None, help="Run DM test vs this benchmark col")
    p.add_argument("--dm_lag", type=int, default=1, help="Forecast horizon 'h' for DM test")
    p.add_argument("--spa_base", type=str, default=None, help="Run SPA test vs this benchmark")
    p.add_argument("--spa_B", type=int, default=500, help="Bootstrap reps for SPA")
    p.add_argument("--mcs", action="store_true", help="Compute Model Confidence Set")
    p.add_argument("--mcs_alpha", type=float, default=0.10, help="Significance level for MCS")
    args = p.parse_args()

    evaluate_all(
        panel_csv=args.panel_csv,
        dm_base=args.dm_base,
        dm_lag=args.dm_lag,
        spa_base=getattr(args, "spa_base", None),
        spa_B=getattr(args, "spa_B", 500),
        mcs=getattr(args, "mcs", False),
        mcs_alpha=getattr(args, "mcs_alpha", 0.10),
    ) 