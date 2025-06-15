from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from econ499.utils import load_config
from econ499.utils.metrics_utils import rmse, mae
from econ499.eval.utils import _load_predictions, _mape, _qlike

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()
ARTIFACT_DIR = ROOT / "artifacts" / "tables"
TARGET_COL = CFG["features"]["target_col"]
VIX_COL = "VIXCLS"


def _build_eval_df(panel_csv: str | Path | None = None) -> pd.DataFrame:
    if panel_csv is None:
        default_path = Path(CFG["paths"]["drl_state_file"]).resolve()
        alt_path = DATA_DIR / "spx_iv_drl_state.csv"
        panel_csv = default_path if default_path.exists() else alt_path
    panel = pd.read_csv(panel_csv, parse_dates=["date"]).sort_values("date")
    panel["IV_next"] = panel[TARGET_COL].shift(-1)
    panel.dropna(subset=["IV_next"], inplace=True)

    eval_df = panel[["date", "IV_next", VIX_COL]].copy()
    eval_df["naive"] = panel[TARGET_COL].values
    eval_df["ar1"] = panel[TARGET_COL].shift(1).rolling(2, min_periods=1).mean().bfill()

    for name, df_pred in _load_predictions():
        eval_df = eval_df.merge(df_pred, on="date", how="left")
    eval_df.dropna(inplace=True)
    return eval_df


def evaluate_by_regime(panel_csv: str | Path | None = None, *, vix_thresh: float = 20.0) -> Path:
    df = _build_eval_df(panel_csv)
    df["regime"] = pd.Series(
        pd.Categorical(np.where(df[VIX_COL] <= vix_thresh, "calm", "turbulent"))
    )

    rows: list[dict] = []
    for regime, subset in df.groupby("regime"):
        y_true = subset["IV_next"].to_numpy()
        naive_mae = mae(y_true, subset["naive"].to_numpy())
        for col in subset.columns:
            if col in {"date", "IV_next", VIX_COL, "regime"}:
                continue
            yhat = subset[col].to_numpy()
            rows.append({
                "regime": regime,
                "model": col,
                "RMSE": rmse(y_true, yhat),
                "MAE": mae(y_true, yhat),
                "MASE": mae(y_true, yhat) / naive_mae,
                "MAPE(%)": _mape(y_true, yhat),
                "QLIKE": _qlike(y_true, yhat),
            })

    metrics = pd.DataFrame(rows)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "forecast_metrics_vix_regimes.csv"
    metrics.to_csv(out_path, index=False, float_format="%.6f")

    for regime in ["calm", "turbulent"]:
        sub = metrics[metrics["regime"] == regime]
        if not sub.empty:
            print(f"\n=== {regime.capitalize()} regime ===")
            print(
                sub.set_index("model")[["RMSE", "MAE", "MASE"]]
                .sort_values("RMSE")
                .to_markdown(floatfmt=".6f")
            )
    print("Saved metrics to", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forecasts by VIX regime")
    parser.add_argument("--panel_csv", type=str, default=None)
    parser.add_argument("--vix_thresh", type=float, default=20.0, help="Threshold for calm vs turbulent")
    args = parser.parse_args()
    evaluate_by_regime(panel_csv=args.panel_csv, vix_thresh=args.vix_thresh)


if __name__ == "__main__":
    main()
