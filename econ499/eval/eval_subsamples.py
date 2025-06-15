from __future__ import annotations

"""Evaluate forecast accuracy across different date ranges."""

from pathlib import Path
from datetime import datetime
import pandas as pd

from econ499.utils import load_config
from econ499.utils.metrics_utils import rmse, mae
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


def evaluate_subsamples(panel_csv: str | Path | None = None, *, years: list[int] | None = None) -> Path:
    """Compute metrics on multiple date sub-samples.

    Parameters
    ----------
    years : list[int], optional
        Start years for each sub-sample. Defaults to ``[2010, 2015, 2020]``.
    """
    df = _build_eval_df(panel_csv)
    if years is None:
        years = [2010, 2015, 2020]

    bounds = years + [df["date"].dt.year.max() + 1]
    records: list[dict] = []

    for start, end in zip(bounds[:-1], bounds[1:]):
        mask = (df["date"].dt.year >= start) & (df["date"].dt.year < end)
        subset = df.loc[mask]
        if subset.empty:
            continue
        y = subset["IV_next"].to_numpy()
        naive_mae = mae(y, subset["naive"].to_numpy())
        label = f"{start}-{end-1}"
        for col in subset.columns:
            if col in {"date", "IV_next"}:
                continue
            yhat = subset[col].to_numpy()
            records.append({
                "period": label,
                "model": col,
                "RMSE": rmse(y, yhat),
                "MAE": mae(y, yhat),
                "MASE": mae(y, yhat) / naive_mae,
                "MAPE(%)": _mape(y, yhat),
                "QLIKE": _qlike(y, yhat),
            })

    out_df = pd.DataFrame(records)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "forecast_metrics_subsamples.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print("Saved sub-sample metrics ->", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate forecasts over sub-samples")
    p.add_argument("--panel_csv", type=str, default=None)
    p.add_argument("--years", nargs="*", type=int, help="Start years e.g. 2010 2015 2020")
    args = p.parse_args()
    evaluate_subsamples(panel_csv=args.panel_csv, years=args.years)
