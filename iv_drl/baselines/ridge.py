from __future__ import annotations

"""Ridge-regularised OLS baseline for ATM-IV forecasting.

Follows Chen, Kim & Lee (2023) who advocate ridge shrinkage when the feature
set is high-dimensional and collinear.
"""

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import rmse, mae
from iv_drl.baselines.ols import _build_X  # reuse helper

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()


# ----------------------------------------------------------------------
# main API
# ----------------------------------------------------------------------

def run_ridge(
    *,
    out_csv: str | Path | None = None,
    alphas: Sequence[float] | None = None,
    train_ratio: float = 0.8,
) -> Path:
    """Fit ridge regression on the full state and export OOS forecasts.

    Parameters
    ----------
    out_csv : str | Path | None
        Destination of forecast CSV. Defaults to ``data_processed/ridge_oos_predictions.csv``.
    alphas : sequence of float, optional
        Ridge penalty grid. Default [1e-3, 1e-2, 0.1, 1.0, 10.0].
    train_ratio : float
        Chronological split between train and OOS.
    """

    alphas = alphas or [1e-3, 1e-2, 0.1, 1.0, 10.0]

    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    df["IV_next"] = df[CFG["features"]["target_col"]].shift(-1)
    df.dropna(subset=["IV_next"], inplace=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    oos_df = df.iloc[split_idx:]

    # Build feature matrix (reuse OLS helper)
    X_train = _build_X(train_df, include_volume=True, include_illiq=True, with_groups=True)
    X_oos = _build_X(oos_df, include_volume=True, include_illiq=True, with_groups=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_oos_scaled = scaler.transform(X_oos)

    model = RidgeCV(alphas=alphas, cv=5, fit_intercept=True).fit(X_train_scaled, train_df["IV_next"])

    preds_oos = model.predict(X_oos_scaled)

    out_path = Path(out_csv) if out_csv else OUT_DIR / "ridge_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": oos_df["date"], "ridge_forecast": preds_oos}).to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )

    # quick metrics
    print(
        "OOS  RMSE {:.6f}  MAE {:.6f}  chosen_alpha {:.4g}".format(
            rmse(oos_df["IV_next"].to_numpy(), preds_oos),
            mae(oos_df["IV_next"].to_numpy(), preds_oos),
            float(model.alpha_),
        )
    )
    print('Saved Ridge forecasts to', out_path)
    return out_path


if __name__ == "__main__":
    run_ridge() 