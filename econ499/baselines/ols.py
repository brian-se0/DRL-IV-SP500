"""OLS benchmark for next-day ATM-IV forecast.

Migrated from legacy *src/ols_iv_forecast.py* and adapted to the new
package-based layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import statsmodels.api as sm

from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import rmse, mae

CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _continuous_cols(df: pd.DataFrame) -> list[str]:
    cats = set(CFG["features"].get("categorical_cols", []))
    return [c for c in CFG["features"]["all_feature_cols"] if c in df.columns and c not in cats]


def _build_X(
    df: pd.DataFrame,
    *,
    include_volume: bool,
    include_illiq: bool,
    with_groups: bool,
) -> pd.DataFrame:
    cols: list[str] = []

    base = _continuous_cols(df)
    cols.extend(base)

    if not include_volume:
        cols = [c for c in cols if "Volume" not in c and "volume" not in c]
    if not include_illiq:
        cols = [c for c in cols if "illiq" not in c]

    X = df[cols].copy() if cols else pd.DataFrame(index=df.index)

    if with_groups:
        cat_cols = [c for c in CFG["features"].get("categorical_cols", []) if c in df.columns]
        if cat_cols:
            dummies = pd.get_dummies(df[cat_cols], drop_first=False, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    return X


# ----------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------

def run_ols(
    *,
    out_csv: str | Path | None = None,
    include_volume: bool = True,
    include_illiq: bool = True,
    with_groups: bool = True,
    train_ratio: float = 0.8,
) -> Path:
    """Fit OLS on historical panel and save OOS forecasts.

    Parameters
    ----------
    out_csv : str | Path | None
        Where to write the CSV of OOS predictions.  Defaults to
        ``data_processed/ols_oos_predictions.csv``.
    include_volume, include_illiq, with_groups
        Feature-selection flags mirroring the old CLI.
    train_ratio : float
        Chronological split between train and OOS.
    """

    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")

    target = CFG["features"]["target_col"]
    df["IV_next"] = df[target].shift(-1)
    df.dropna(subset=["IV_next"], inplace=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    oos_df = df.iloc[split_idx:]

    X_train = _build_X(
        train_df,
        include_volume=include_volume,
        include_illiq=include_illiq,
        with_groups=with_groups,
    )
    y_train = train_df["IV_next"]

    # Add / skip intercept according to group dummies choice
    if X_train.shape[1] == 0 or not with_groups:
        X_train = sm.add_constant(X_train, has_constant="add")
    else:
        X_train = sm.add_constant(X_train, has_constant="skip")

    model = sm.OLS(y_train, X_train).fit()

    def _predict(df_slice: pd.DataFrame) -> pd.Series:
        X = _build_X(
            df_slice,
            include_volume=include_volume,
            include_illiq=include_illiq,
            with_groups=with_groups,
        )
        if X.shape[1] == 0 or not with_groups:
            X = sm.add_constant(X, has_constant="add")
        else:
            X = sm.add_constant(X, has_constant="skip")
        return model.predict(X)

    df["ols_forecast"] = _predict(df)
    out_path = Path(out_csv) if out_csv else OUT_DIR / "ols_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[oos_df.index, ["date", "ols_forecast"]].to_csv(out_path, index=False, date_format="%Y-%m-%d")

    # Log metrics for quick sanity-check
    print(
        "TRAIN   RMSE {:.6f}   MAE {:.6f}".format(
            rmse(y_train, df.loc[train_df.index, "ols_forecast"]),
            mae(y_train, df.loc[train_df.index, "ols_forecast"]),
        )
    )
    print('Saved OLS forecasts to', out_path)
    return out_path


if __name__ == "__main__":
    run_ols() 