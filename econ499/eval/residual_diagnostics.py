from __future__ import annotations

"""Residual diagnostics for DRL forecasts using Ljung-Box tests."""

from pathlib import Path
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

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


def check_residuals(panel_csv: str | Path | None = None, *, lags: int = 5) -> Path:
    df = _build_eval_df(panel_csv)
    y = df["IV_next"].to_numpy()

    rows = []
    for col in df.columns:
        if col in {"date", "IV_next"}:
            continue
        resid = df[col].to_numpy() - y
        lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
        rows.append({
            "model": col,
            "RMSE": rmse(y, df[col].to_numpy()),
            "LjungBoxQ": lb["lb_stat"].iloc[0],
            "p_value": lb["lb_pvalue"].iloc[0],
        })

    out_df = pd.DataFrame(rows).sort_values("RMSE")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / "forecast_residual_lb.csv"
    out_df.to_csv(out_path, index=False, float_format="%.6f")
    print("Saved residual diagnostics ->", out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Ljung-Box diagnostics for forecast residuals")
    p.add_argument("--panel_csv", type=str, default=None)
    p.add_argument("--lags", type=int, default=5)
    args = p.parse_args()
    check_residuals(panel_csv=args.panel_csv, lags=args.lags)
