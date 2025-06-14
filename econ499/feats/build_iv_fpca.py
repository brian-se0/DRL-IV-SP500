import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from econ499.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CONFIG = load_config("data_config.yaml")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _nearest_iv(df: pd.DataFrame, ttm_days: int, target_delta: float, is_call: bool) -> float:
    """Return IV for the option closest to *target_delta* within ±5 days of *ttm_days*.

    Parameters
    ----------
    df : pd.DataFrame (single date slice)
    ttm_days : int
        Target time-to-maturity pivot (e.g., 30, 60, 90, 180).
    target_delta : float
        Delta level in *absolute* terms (0.25 or 0.50).
    is_call : bool
        Select calls if True, puts otherwise.
    """
    mask = df["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    if is_call:
        sub = df.loc[mask & (df["option_type"] == "C")].copy()
    else:
        sub = df.loc[mask & (df["option_type"] == "P")].copy()

    if sub.empty:
        return np.nan

    # distance of |delta - target|
    sub["delta_dist"] = (sub["delta_1545"].abs() - target_delta).abs()
    return sub.loc[sub["delta_dist"].idxmin()]["implied_volatility_1545"]


def _build_daily_row(df_day: pd.DataFrame, maturities: List[int]) -> Dict:
    """Extract 12-node IV grid for one trading day."""
    out: Dict[str, float] = {}
    for ttm in maturities:
        out[f"iv_{ttm}d_put25d"] = _nearest_iv(df_day, ttm, 0.25, is_call=False)
        out[f"iv_{ttm}d_atm"] = _nearest_iv(df_day, ttm, 0.50, is_call=True)
        out[f"iv_{ttm}d_call25d"] = _nearest_iv(df_day, ttm, 0.25, is_call=True)
    return out

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def calculate_fpca_factors(df: pd.DataFrame, maturities: List[int] | None = None) -> pd.DataFrame:
    """Return daily FPCA factor scores (3 components) from raw option rows."""

    if df.empty:
        return pd.DataFrame()

    maturities = maturities or [30, 60, 90, 180]

    logging.info("Pre-processing option rows…")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ttm_days"] = (df["expiration"] - df["quote_date"]).dt.days

    # Keep only rows with necessary fields
    req = ["ttm_days", "delta_1545", "implied_volatility_1545"]
    df.dropna(subset=req, inplace=True)
    df = df[df["ttm_days"] > 0]

    # ------------------------------------------------------------------
    # Build daily 12-node grid
    # ------------------------------------------------------------------
    rows = []
    for date, df_day in tqdm(df.groupby("quote_date"), desc="Daily IV grid"):
        row = _build_daily_row(df_day, maturities)
        row["date"] = date
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    grid = pd.DataFrame(rows).set_index("date").sort_index()

    # Drop days with any missing node (strict for PCA)
    grid.dropna(inplace=True)
    if grid.empty:
        logging.warning("All days have missing IV nodes; cannot compute FPCA.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # FPCA (here a plain PCA because data are on a finite grid)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(grid.values)

    pca = PCA(n_components=3)
    factors = pca.fit_transform(X_scaled)

    factors_df = pd.DataFrame(
        factors,
        index=grid.index,
        columns=["iv_fpca1", "iv_fpca2", "iv_fpca3"],
    )

    logging.info(
        "FPCA explained variance ratios: %s",
        np.round(pca.explained_variance_ratio_, 3).tolist(),
    )
    return factors_df


def main():
    input_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "iv_fpca_factors.parquet"
    
    # Skip if output file already exists
    if output_path.exists():
        logging.info('FPCA factors already exist -> %s', output_path)
        return
        
    if not input_dir.exists():
        logging.error("OptionMetrics parquet directory not found: %s", input_dir)
        return

    # Search for parquet files in all year subdirectories
    files = sorted(list(input_dir.glob("**/*.parquet")))
    if not files:
        logging.error("No parquet files found in %s or its subdirectories", input_dir)
        return

    logging.info("Found %d parquet files", len(files))
    
    # Process files in batches by year
    all_features = []
    current_year = None
    year_files = []
    
    for file in tqdm(files, desc="Processing files"):
        year = file.parent.name
        if year != current_year:
            # Process previous year's batch
            if year_files:
                logging.info("Processing %d files from year %s", len(year_files), current_year)
                year_data = pd.concat((pd.read_parquet(f) for f in year_files), ignore_index=True)
                year_features = calculate_fpca_factors(year_data)
                if not year_features.empty:
                    all_features.append(year_features)
            # Start new batch
            current_year = year
            year_files = []
        year_files.append(file)
    
    # Process final batch
    if year_files:
        logging.info("Processing %d files from year %s", len(year_files), current_year)
        year_data = pd.concat((pd.read_parquet(f) for f in year_files), ignore_index=True)
        year_features = calculate_fpca_factors(year_data)
        if not year_features.empty:
            all_features.append(year_features)
    
    if not all_features:
        logging.warning("No FPCA factors generated.")
        return
        
    # Combine all features
    factors_df = pd.concat(all_features)
    factors_df.to_parquet(output_path)
    logging.info('Saved FPCA factors -> %s', output_path)


if __name__ == "__main__":
    main() 