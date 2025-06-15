from __future__ import annotations

"""Evaluate forecast robustness across multiple random seeds."""

from pathlib import Path
import re
import pandas as pd
import numpy as np

from econ499.utils import load_config
from econ499.utils.metrics_utils import rmse, mae
from econ499.eval.utils import _mape, _qlike
from econ499.eval.evaluate_all import _find_forecast_col

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
    return eval_df


def evaluate_multi_seed(pattern: str = "*_seed*_oos_predictions.csv", *, panel_csv: str | Path | None = None) -> Path:
    """Aggregate forecast metrics across multiple random seeds.

    Parameters
    ----------
    pattern : str, default "*_seed*_oos_predictions.csv"
        Glob pattern used to locate prediction files in ``DATA_DIR``.
    panel_csv : str or Path, optional
        Optional override for the evaluation panel CSV.
    """
    base_eval = _build_eval_df(panel_csv)

    files = sorted(DATA_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matching {pattern}")

    records: list[dict] = []
    for csv in files:
        df_pred = pd.read_csv(csv, parse_dates=["date"])
        col = _find_forecast_col(df_pred)
        run_name = Path(csv).stem.replace("_oos_predictions", "")
        algo = re.sub(r"_seed\d+", "", run_name)
        merged = base_eval.merge(df_pred[["date", col]].rename(columns={col: run_name}), on="date", how="left")
        merged.dropna(inplace=True)
        y = merged["IV_next"].to_numpy()
        yhat = merged[run_name].to_numpy()
        naive_mae = mae(y, merged["naive"].to_numpy())
        records.append({
            "algo": algo,
            "run": run_name,
            "RMSE": rmse(y, yhat),
            "MAE": mae(y, yhat),
            "MASE": mae(y, yhat) / naive_mae,
            "MAPE(%)": _mape(y, yhat),
            "QLIKE": _qlike(y, yhat),
        })

    metrics = pd.DataFrame(records)
    summary = metrics.groupby("algo")[["RMSE", "MAE", "MASE", "MAPE(%)", "QLIKE"]].agg(["mean", "std"])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_metrics = ARTIFACT_DIR / "seed_run_metrics.csv"
    out_summary = ARTIFACT_DIR / "seed_run_summary.csv"
    metrics.to_csv(out_metrics, index=False, float_format="%.6f")
    summary.to_csv(out_summary, float_format="%.6f")

    print("Saved per-run metrics ->", out_metrics)
    print("Saved summary ->", out_summary)
    return out_summary


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate multiple seed runs")
    p.add_argument("--pattern", type=str, default="*_seed*_oos_predictions.csv", help="Glob for forecast files")
    p.add_argument("--panel_csv", type=str, default=None)
    args = p.parse_args()
    evaluate_multi_seed(pattern=args.pattern, panel_csv=args.panel_csv)
