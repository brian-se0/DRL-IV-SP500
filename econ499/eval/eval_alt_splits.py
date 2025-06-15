from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import rmse, mae
from econ499.eval.utils import _load_predictions, _mape, _qlike

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()
ARTIFACT_DIR = ROOT / "artifacts" / "tables"
TARGET_COL = CFG["features"]["target_col"]


def _build_eval_df(panel_csv: str | Path | None = None) -> pd.DataFrame:
    if panel_csv is None:
        default_path = Path(CFG["paths"]["drl_state_file"]).resolve()
        alt_path = DATA_DIR / "spx_iv_drl_state.csv"
        panel_csv = default_path if default_path.exists() else alt_path
    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    panel["IV_next"] = panel[TARGET_COL].shift(-1)
    panel.dropna(subset=["IV_next"], inplace=True)

    eval_df = panel[["date", "IV_next"]].copy()
    eval_df["naive"] = panel[TARGET_COL].values
    eval_df["ar1"] = panel[TARGET_COL].shift(1).rolling(2, min_periods=1).mean().bfill()

    for name, df_pred in _load_predictions():
        eval_df = eval_df.merge(df_pred, on="date", how="left")
    eval_df.dropna(inplace=True)
    return eval_df


def evaluate_alt_splits(panel_csv: str | Path | None = None,
                         offsets: list[int] | None = None) -> Path:
    """Compute metrics for different OOS start dates (year offsets).

    Parameters
    ----------
    offsets : list[int]
        Year offsets relative to the default split (70% quantile date). If None,
        uses [-5, -3, 0, +3, +5].
    """
    df = _build_eval_df(panel_csv)

    default_idx = int(0.7 * len(df))
    default_date = df.iloc[default_idx]["date"]

    if offsets is None:
        offsets = [-5, -3, 0, 3, 5]

    rows = []
    for off in offsets:
        start_date = default_date + timedelta(days=365 * off)
        subset = df[df["date"] >= start_date]
        if subset.empty:
            continue
        y_true = subset["IV_next"].to_numpy()
        naive_mae = mae(y_true, subset["naive"].to_numpy())
        for col in subset.columns:
            if col in {"date", "IV_next"}:
                continue
            yhat = subset[col].to_numpy()
            rows.append({
                "offset_years": off,
                "model": col,
                "RMSE": rmse(y_true, yhat),
                "MAE": mae(y_true, yhat),
                "MASE": mae(y_true, yhat) / naive_mae,
                "MAPE(%)": _mape(y_true, yhat),
                "QLIKE": _qlike(y_true, yhat),
            })

    df_metrics = pd.DataFrame(rows)
    out_path = ARTIFACT_DIR / "forecast_metrics_alt_splits.csv"
    df_metrics.to_csv(out_path, index=False, float_format="%.6f")
    print('Saved', out_path)
    # also print summary per offset
    for off in offsets:
        sub = df_metrics[df_metrics["offset_years"] == off].copy()
        if sub.empty:
            continue
        print(f"\n=== Offset {off:+} years (start date >= {(default_date + timedelta(days=365*off)).date()}) ===")
        print(
            sub.set_index("model")[["RMSE", "MAE", "MASE"]]
            .sort_values("RMSE")
            .to_markdown(floatfmt=".6f")
        )
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv", type=str, default=None)
    p.add_argument("--offsets", nargs="*", type=int, help="Year offsets e.g. -5 0 5")
    args = p.parse_args()

    evaluate_alt_splits(panel_csv=args.panel_csv, offsets=args.offsets) 