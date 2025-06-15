from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from econ499.utils import load_config
from econ499.utils.metrics_utils import rmse, mae
from econ499.eval.evaluate_all import (
    _load_predictions,
    _find_forecast_col,
    _mape,
    _qlike,
)

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()
ARTIFACT_DIR = ROOT / "artifacts" / "tables"
TARGET_COL = CFG["features"]["target_col"]


def _build_eval_df(panel_csv: Path | str | None) -> pd.DataFrame:
    if panel_csv is None:
        default_path = Path(CFG["paths"]["drl_state_file"]).resolve()
        alt_path = DATA_DIR / "spx_iv_drl_state.csv"
        panel_csv = default_path if default_path.exists() else alt_path
    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    panel["IV_next"] = panel[TARGET_COL].shift(-1)
    panel.dropna(subset=["IV_next"], inplace=True)

    eval_df = panel[["date", "IV_next"]].copy()
    eval_df["naive"] = panel[TARGET_COL].values
    eval_df["ar1"] = (
        panel[TARGET_COL].shift(1).rolling(2, min_periods=1).mean().bfill()
    )

    for name, df_pred in _load_predictions():
        eval_df = eval_df.merge(df_pred, on="date", how="left")

    eval_df.dropna(inplace=True)
    return eval_df


def walk_forward(panel_csv: str | Path | None = None,
                  *,
                  init_years: int = 5,
                  step_days: int = 20) -> Path:
    """Walk-forward evaluation with expanding window.

    Parameters
    ----------
    init_years : int
        Size of the initial training window in calendar years.
    step_days : int
        Re-evaluation frequency (in trading days).
    """
    df = _build_eval_df(panel_csv)

    start_date = df["date"].min() + timedelta(days=365 * init_years)
    mask = df["date"] >= start_date
    if not mask.any():
        # not enough history → start at first available point
        idx0 = 0
    else:
        idx0 = mask.idxmax()

    windows = []  # list of (start_idx,end_idxInclusive)
    cur = idx0
    while cur < len(df):
        end_idx = min(cur + step_days - 1, len(df) - 1)
        windows.append((cur, end_idx))
        cur += step_days

    records: list[dict] = []
    for w_start, w_end in windows:
        slice_df = df.iloc[w_start : w_end + 1]
        y_true = slice_df["IV_next"].to_numpy()
        naive_mae = mae(y_true, slice_df["naive"].to_numpy())
        for col in slice_df.columns:
            if col in {"date", "IV_next"}:
                continue
            yhat = slice_df[col].to_numpy()
            rec = {
                "window_start": slice_df.iloc[0]["date"],
                "window_end": slice_df.iloc[-1]["date"],
                "model": col,
                "RMSE": rmse(y_true, yhat),
                "MAE": mae(y_true, yhat),
            }
            rec["MASE"] = rec["MAE"] / naive_mae
            rec["MAPE(%)"] = _mape(y_true, yhat)
            rec["QLIKE"] = _qlike(y_true, yhat)
            records.append(rec)

    metrics_long = pd.DataFrame(records)
    summary = (
        metrics_long.groupby("model")[["RMSE", "MAE", "MASE", "MAPE(%)", "QLIKE"]]
        .mean()
        .sort_values("RMSE")
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "forecast_metrics_walk.csv"
    summary.to_csv(out_path, float_format="%.6f")
    print(summary.to_markdown(floatfmt=".6f"))
    print('Walk-forward metrics saved to', out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv", type=str, default=None)
    p.add_argument("--init_years", type=int, default=5)
    p.add_argument("--step_days", type=int, default=20)
    args = p.parse_args()

    walk_forward(panel_csv=args.panel_csv, init_years=args.init_years, step_days=args.step_days) 