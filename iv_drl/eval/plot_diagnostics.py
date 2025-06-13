from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import rmse

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()
FIG_DIR = ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TARGET_COL = CFG["features"]["target_col"]


# -----------------------------------------------------------------------------
# Helpers to load the evaluation panel and forecasts (same rules as evaluate_all)
# -----------------------------------------------------------------------------

def _find_forecast_col(df: pd.DataFrame) -> str:
    others = [c for c in df.columns if c.lower() != "date"]
    if len(others) == 1:
        return others[0]
    for c in others:
        if c.endswith("_forecast"):
            return c
    raise ValueError("Unable to identify forecast column")


def _load_forecasts() -> list[tuple[str, pd.Series]]:
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

    for name, df_pred in _load_forecasts():
        eval_df = eval_df.merge(df_pred, on="date", how="left")
    eval_df.dropna(inplace=True)
    return eval_df


# -----------------------------------------------------------------------------
# Plotting routines
# -----------------------------------------------------------------------------

def _plot_one_model(eval_df: pd.DataFrame, model: str):
    dates = eval_df["date"]
    y = eval_df["IV_next"].to_numpy()
    yhat = eval_df[model].to_numpy()
    residuals = yhat - y

    rmse_val = rmse(y, yhat)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Diagnostics – {model} (RMSE={rmse_val:.4f})", fontsize=14)

    # 1. Time-series overlay
    ax = axes[0, 0]
    ax.plot(dates, y, label="Actual", linewidth=1)
    ax.plot(dates, yhat, label="Forecast", linewidth=1)
    ax.set_title("Actual vs Forecast")
    ax.legend()

    # 2. Scatter + OLS fit
    ax = axes[0, 1]
    sns.regplot(x=y, y=yhat, ax=ax, scatter_kws={"s": 8, "alpha": 0.5}, line_kws={"color": "red"})
    ax.set_xlabel("Actual")
    ax.set_ylabel("Forecast")
    ax.set_title("Scatter & OLS fit")

    # 3. Residual histogram
    ax = axes[1, 0]
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_title("Residuals histogram")

    # 4. Q–Q plot
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q–Q plot of residuals")

    # 5. Rolling 250-day RMSE
    ax = axes[2, 0]
    window = 250
    roll_rmse = (
        pd.Series((y - yhat) ** 2).rolling(window).mean().apply(np.sqrt)
    )
    ax.plot(dates, roll_rmse)
    ax.set_title("Rolling 250-day RMSE")

    # 6. Empty / watermark
    axes[2, 1].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = FIG_DIR / f"diag_{model}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('Saved', out_path)


def plot_all(panel_csv: str | Path | None = None, models: list[str] | None = None):
    eval_df = _build_eval_df(panel_csv)

    if models is None:
        models = [c for c in eval_df.columns if c not in {"date", "IV_next"}]

    for m in models:
        _plot_one_model(eval_df, m)

    return FIG_DIR


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--panel_csv", type=str, default=None)
    p.add_argument("--models", nargs="*", help="Subset of model names to plot (default: all)")
    args = p.parse_args()

    plot_all(panel_csv=args.panel_csv, models=args.models) 