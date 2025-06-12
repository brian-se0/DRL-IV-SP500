from __future__ import annotations

"""Helper script to build publication-ready tables and figures.

Usage:
    python scripts/generate_pub_assets.py

Outputs:
    • artifacts/tables/forecast_metrics.tex       – standalone LaTeX table
    • artifacts/figures/diag_<model>.png          – diagnostics for each model

The script simply wraps the existing evaluation & plotting helpers so that a
single command refreshes everything after a new rebuild.
"""

from pathlib import Path
import pandas as pd
import sys

# ensure project root is on PYTHONPATH when script called via absolute path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iv_drl.evaluation.evaluate_all import evaluate_all
from iv_drl.evaluation.plot_diagnostics import plot_all

# -----------------------------------------------------------------------------
# 1. LaTeX table
# -----------------------------------------------------------------------------
out_csv = evaluate_all(dm_base=None)  # reuse cached metrics if already exists
metrics_df = pd.read_csv(out_csv)
tex_path = Path(out_csv).with_suffix('.tex')
tex_path.write_text(metrics_df.to_latex(index=False, float_format='%.4f', caption='Forecast accuracy comparison', label='tab:forecast_metrics'))
print('LaTeX table saved to', tex_path)

# -----------------------------------------------------------------------------
# 2. Diagnostic plots
# -----------------------------------------------------------------------------

# determine representative model names for plots
def _pick(model_prefix: str) -> str:
    """Return the first column name starting with prefix (prefers exact match)."""
    cols = metrics_df['model'].tolist()
    if model_prefix in cols:
        return model_prefix
    # fallback: smallest RMSE among prefix_* variants
    variants = [c for c in cols if c.startswith(model_prefix)]
    if not variants:
        return model_prefix  # let plot_all raise if truly missing
    best = metrics_df.loc[metrics_df['model'].isin(variants)].sort_values('RMSE').iloc[0]['model']
    return best

ppo_name = _pick('ppo')
a2c_name = _pick('a2c')

plot_all(models=[ppo_name, a2c_name, 'har_rv'])

print('Diagnostic plots refreshed') 