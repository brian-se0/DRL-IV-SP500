from __future__ import annotations
"""Build the two high-level figures for the paper.

1. Diebold–Mariano p-value heat-map comparing every forecasting model.
2. Model-Confidence-Set (MCS) size bar chart across α ∈ {0.10,0.05,0.01}.

The script expects the consolidated metrics table generated by
`evaluation/evaluate_all.py` to live at
``artifacts/tables/forecast_metrics.csv``.
Figures are saved to ``artifacts/paper_figs/``.

Run:
    python analysis/make_paper_figs.py

Dependencies: pandas, numpy, matplotlib, seaborn, statsmodels.
"""

from pathlib import Path
import itertools
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_lm
from statsmodels.stats.weightstats import ztest
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import ccf
from scipy import stats
import yaml

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_CSV = PROJECT_ROOT / "artifacts" / "tables" / "forecast_metrics.csv"
CFG = yaml.safe_load((PROJECT_ROOT / "cfg" / "data_config.yaml").read_text())
STATE_CSV = (PROJECT_ROOT / CFG["paths"]["drl_state_file"]).resolve()
FORECAST_DIR = (PROJECT_ROOT / CFG["paths"]["output_dir"]).resolve()
OUT_DIR = PROJECT_ROOT / "artifacts" / "paper_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Which error column to run DM test on
ERROR_COL = "rmse"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> float:
    """Return Diebold-Mariano two-sided p-value for equal forecast accuracy.

    Very light-weight implementation with Newey–West variance correction
    (lag = h-1).  Assumes mean-squared-error loss.
    """

    d_t = e1 ** 2 - e2 ** 2  # loss differential
    mean_d = np.mean(d_t)
    # Newey–West estimator of var(d_t)
    n = len(d_t)
    gamma0 = np.var(d_t, ddof=1)
    var = gamma0
    for lag in range(1, h):
        gamma = np.cov(d_t[lag:], d_t[:-lag], ddof=1)[0, 1]
        var += 2 * (1 - lag / h) * gamma
    # Guard against zero variance or tiny sample size
    if n <= 1 or var == 0 or np.isnan(var):
        return np.nan

    dm_stat = mean_d / np.sqrt(var / n)
    pval = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return pval


def _compute_dm_matrix(df: pd.DataFrame, model_cols: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(index=model_cols, columns=model_cols, dtype=float)
    for m1, m2 in itertools.combinations(model_cols, 2):
        p = _dm_test(df[m1] - df["iv_t_plus1"], df[m2] - df["iv_t_plus1"], h=1)
        mat.at[m1, m2] = p
        mat.at[m2, m1] = p
    np.fill_diagonal(mat.values, np.nan)
    return mat


def _plot_dm_heatmap(pmat: pd.DataFrame, path: Path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(pmat, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={"label": "p-value"})
    plt.title("Diebold–Mariano p-values")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _plot_mcs_bar(models: List[str], sizes: List[int], path: Path):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=sizes, color="#4c72b0")
    plt.ylabel("Models retained")
    plt.title("Model-Confidence-Set size (α = 0.05)")
    plt.ylim(0, max(sizes) + 1)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    if not METRIC_CSV.exists():
        raise FileNotFoundError(f"Metrics table not found: {METRIC_CSV}")

    # Build DM matrix from raw forecast CSVs
    # --------------------------------------------------------------
    forecast_files = list(FORECAST_DIR.glob("*_oos_predictions.csv"))
    if not forecast_files:
        warnings.warn(f"No *_oos_predictions.csv found in {FORECAST_DIR}; skipping DM heat-map.")
    else:
        # Load true series
        state_df = pd.read_csv(STATE_CSV, parse_dates=["date"])
        target_col = CFG["features"]["target_col"]
        state_df = state_df.sort_values("date")
        state_df["iv_t_plus1"] = state_df[target_col].shift(-1)
        true_series = state_df[["date", "iv_t_plus1"]]

        merged: pd.DataFrame = true_series.copy()

        model_cols: List[str] = []
        for fp in forecast_files:
            df_f = pd.read_csv(fp, parse_dates=["date"])
            col = [c for c in df_f.columns if c != "date"][0]
            merged = merged.merge(df_f, on="date", how="inner")
            model_cols.append(col)

        # Drop any rows with NaNs
        merged.dropna(subset=["iv_t_plus1", *model_cols], inplace=True)

        if len(model_cols) < 2:
            warnings.warn("Need at least two forecast columns for DM test; skipping.")
        else:
            pmat = _compute_dm_matrix(merged, model_cols)
            # Replace infinities or invalid entries
            pmat.replace([np.inf, -np.inf], np.nan, inplace=True)
            _plot_dm_heatmap(pmat, OUT_DIR / "fig_dm_heatmap.png")
            print("[FIG] DM heat-map →", OUT_DIR / "fig_dm_heatmap.png")

    # Refresh metrics DataFrame (may have been unloaded in memory-saving paths)
    metrics = pd.read_csv(METRIC_CSV)

    # ----- Figure 2: MCS bar chart -----
    if "mcs_in_set" in metrics.columns:
        mcs = metrics.groupby("model")["mcs_in_set"].max().astype(int)
        _plot_mcs_bar(mcs.index.tolist(), mcs.tolist(), OUT_DIR / "fig_mcs_size.png")
        print("[FIG] MCS bar chart →", OUT_DIR / "fig_mcs_size.png")
    else:
        warnings.warn("Column 'mcs_in_set' not found; run evaluator with --mcs to compute the set.")

if __name__ == "__main__":
    main() 