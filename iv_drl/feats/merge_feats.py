"""Merge all feature tables into a single state representation.

This script reads the daily price features, IV surface features and FPCA
factors from the ``results`` directory, aligns them on the common trading
calendar and performs a light round of missing value imputation.  The
resulting dataframe is written to ``spx_iv_drl_state.csv`` and forms the
input for all modelling steps.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from iv_drl.feats.imputation import kalman_impute, volatility_aware_impute
from iv_drl.utils import load_config


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CONFIG = load_config("data_config.yaml")
_cfg_out = Path(CONFIG["paths"]["output_dir"])
OUTPUT_DIR = _cfg_out if _cfg_out.is_dir() else Path("results")

PRICE_PATH = OUTPUT_DIR / "spx_daily_features.parquet"
IV_PATH = OUTPUT_DIR / "iv_surface_daily_features.parquet"
FPCA_PATH = OUTPUT_DIR / "iv_fpca_factors.parquet"
OUT_CSV = OUTPUT_DIR / "spx_iv_drl_state.csv"


def _impute_price(df: pd.DataFrame) -> pd.DataFrame:
    """Impute price related features using a simple Kalman smoother."""
    for col in df.columns:
        df[col] = kalman_impute(df[col])
    return df


def _impute_iv(df: pd.DataFrame) -> pd.DataFrame:
    """Impute IV surface features using volatility aware smoothing."""
    return volatility_aware_impute(df)


def _impute_fpca(df: pd.DataFrame) -> pd.DataFrame:
    """Forward/back fill FPCA factors."""
    return df.ffill().bfill()


def main() -> None:
    """Merge all feature sets and write the consolidated CSV."""
    if not PRICE_PATH.exists() or not IV_PATH.exists() or not FPCA_PATH.exists():
        missing = [p for p in (PRICE_PATH, IV_PATH, FPCA_PATH) if not p.exists()]
        raise FileNotFoundError(f"Missing required feature file(s): {missing}")

    price = pd.read_parquet(PRICE_PATH)
    iv = pd.read_parquet(IV_PATH)
    fpca = pd.read_parquet(FPCA_PATH)

    # Determine common date range
    start = max(price.index.min(), iv.index.min(), fpca.index.min())
    end = min(price.index.max(), iv.index.max(), fpca.index.max())

    price = price.loc[start:end]
    iv = iv.loc[start:end]
    fpca = fpca.loc[start:end]

    price = _impute_price(price)
    iv = _impute_iv(iv)
    fpca = _impute_fpca(fpca)

    df = price.join(iv, how="inner").join(fpca, how="inner")

    # Drop the monotonically increasing integer column produced during
    # earlier processing.  Keeping ``quote_date`` alongside the index
    # matches the expected 36 feature columns.
    if "index" in df.columns:
        df = df.drop(columns=["index"])

    df.to_csv(OUT_CSV)
    logging.info("Saved merged state -> %s", OUT_CSV)


if __name__ == "__main__":  # pragma: no cover
    main()
