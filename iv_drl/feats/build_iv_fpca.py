import logging
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_data(df: pd.DataFrame) -> bool:
    """Validate required columns and basic data quality."""
    if df.empty:
        logging.error("Empty dataframe provided")
        return False
        
    required_cols = ["ttm_days", "delta_1545", "implied_volatility_1545"]
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing required columns: {[col for col in required_cols if col not in df.columns]}")
        return False
        
    return True

def impute_missing_values(series: pd.Series, window: int = 5) -> pd.Series:
    """Impute missing values using rolling mean."""
    return series.fillna(series.rolling(window=window, min_periods=1).mean())

def check_data_quality(df: pd.DataFrame, stage: str) -> None:
    """Log data quality metrics at different processing stages."""
    logging.info(f"\nData quality check at {stage}:")
    
    # Check for missing values
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df)) * 100
            logging.warning(f"{col}: {missing_count} missing values ({missing_pct:.2f}%)")
    
    # Check for outliers using IQR method
    for col in df.select_dtypes(include=[np.number]).columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
        if outliers > 0:
            logging.warning(f"{col}: {outliers} potential outliers detected")

def _nearest_iv(df: pd.DataFrame, ttm_days: int, target_delta: float, is_call: bool) -> float:
    """Find IV for option closest to target delta within ±5 days of target TTM."""
    mask = df["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    option_type = "C" if is_call else "P"
    sub = df.loc[mask & (df["option_type"] == option_type)].copy()

    if sub.empty:
        return np.nan

    sub["delta_dist"] = (sub["delta_1545"].abs() - target_delta).abs()
    return sub.loc[sub["delta_dist"].idxmin()]["implied_volatility_1545"]

def _build_daily_row(df_day: pd.DataFrame, maturities: List[int]) -> Dict[str, float]:
    """Extract IV grid for one trading day with specified maturities."""
    out = {}
    for ttm in maturities:
        out[f"iv_{ttm}d_put25d"] = _nearest_iv(df_day, ttm, 0.25, is_call=False)
        out[f"iv_{ttm}d_atm"] = _nearest_iv(df_day, ttm, 0.50, is_call=True)
        out[f"iv_{ttm}d_call25d"] = _nearest_iv(df_day, ttm, 0.25, is_call=True)
    return out

def analyze_date_ranges(df: pd.DataFrame, name: str) -> None:
    """Analyze and log date range information for a dataframe."""
    if df.empty:
        logging.error(f"{name}: Empty dataframe")
        return
        
    dates = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['date'])
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    actual_days = len(dates.unique())
    missing_days = total_days - actual_days
    
    logging.info(f"\n{name} Date Range Analysis:")
    logging.info(f"Date range: {min_date.date()} to {max_date.date()}")
    logging.info(f"Total calendar days: {total_days}")
    logging.info(f"Actual trading days: {actual_days}")
    logging.info(f"Missing days: {missing_days} ({missing_days/total_days*100:.1f}%)")
    
    # Analyze gaps
    dates_sorted = sorted(dates.unique())
    gaps = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
    if gaps:
        logging.info(f"Average gap between dates: {np.mean(gaps):.1f} days")
        logging.info(f"Maximum gap: {max(gaps)} days")
        logging.info(f"Gaps > 5 days: {sum(1 for g in gaps if g > 5)}")

def analyze_feature_overlap(dfs: Dict[str, pd.DataFrame]) -> None:
    """Analyze overlap between different feature sets."""
    all_dates = set()
    date_sets = {}
    
    for name, df in dfs.items():
        dates = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['date'])
        date_sets[name] = set(dates)
        all_dates.update(dates)
    
    logging.info("\nFeature Set Overlap Analysis:")
    for name1, dates1 in date_sets.items():
        for name2, dates2 in date_sets.items():
            if name1 < name2:  # Avoid duplicate comparisons
                overlap = len(dates1.intersection(dates2))
                total = len(dates1.union(dates2))
                logging.info(f"{name1} vs {name2}:")
                logging.info(f"  Overlapping dates: {overlap}")
                logging.info(f"  Total unique dates: {total}")
                logging.info(f"  Overlap percentage: {overlap/total*100:.1f}%")

def calculate_fpca_factors(df: pd.DataFrame, maturities: List[int] = None) -> pd.DataFrame:
    """Calculate FPCA factors from option data.
    
    Args:
        df: DataFrame containing option data
        maturities: List of target maturities in days (default: [30, 60, 90, 180])
    
    Returns:
        DataFrame with FPCA factor scores
    """
    if df.empty:
        logging.error("Empty dataframe provided")
        return pd.DataFrame()

    # Calculate time to maturity if not present
    if 'ttm_days' not in df.columns:
        if 'quote_date' not in df.columns or 'expiration' not in df.columns:
            logging.error("Missing required columns: quote_date and expiration")
            return pd.DataFrame()
        df['ttm_days'] = (df['expiration'] - df['quote_date']).dt.days

    if not validate_data(df):
        return pd.DataFrame()

    maturities = maturities or [30, 60, 90, 180]

    # Pre-process data
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    
    # Analyze initial date distribution
    analyze_date_ranges(df, "Raw Option Data")
    
    # Log initial shape
    initial_shape = df.shape
    logging.info(f"Initial dataframe shape: {initial_shape}")
    
    # Drop rows with missing values in critical columns
    df.dropna(subset=["ttm_days", "delta_1545", "implied_volatility_1545"], inplace=True)
    after_dropna_shape = df.shape
    logging.info(f"Rows dropped due to missing values in critical columns: {initial_shape[0] - after_dropna_shape[0]}")
    
    # Drop rows with non-positive TTM
    df = df[df["ttm_days"] > 0]
    after_ttm_shape = df.shape
    logging.info(f"Rows dropped due to non-positive TTM: {after_dropna_shape[0] - after_ttm_shape[0]}")

    check_data_quality(df, "after initial cleaning")

    # Build daily IV grid
    rows = []
    for date, df_day in tqdm(df.groupby("quote_date"), desc="Daily IV grid"):
        row = _build_daily_row(df_day, maturities)
        row["date"] = date
        rows.append(row)

    if not rows:
        logging.error("No valid rows generated for IV grid")
        return pd.DataFrame()

    grid = pd.DataFrame(rows).set_index("date").sort_index()
    logging.info(f"IV grid shape after construction: {grid.shape}")
    
    # Analyze date distribution after grid construction
    analyze_date_ranges(grid, "IV Grid")
    
    check_data_quality(grid, "after grid construction")

    # Impute missing values
    for col in grid.columns:
        grid[col] = impute_missing_values(grid[col])

    if grid.isna().any().any():
        logging.warning("Some IV nodes still have missing values after imputation.")
        grid.dropna(inplace=True)
        logging.info(f"Rows dropped after imputation: {grid.shape[0]}")
        if grid.empty:
            logging.error("All days have missing IV nodes after imputation")
            return pd.DataFrame()

    check_data_quality(grid, "after imputation")

    # Perform FPCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(grid.values)

    pca = PCA(n_components=3)
    factors = pca.fit_transform(X_scaled)

    factors_df = pd.DataFrame(
        factors,
        index=grid.index,
        columns=["iv_fpca1", "iv_fpca2", "iv_fpca3"],
    )

    # Analyze final date distribution
    analyze_date_ranges(factors_df, "Final FPCA Factors")

    logging.info(
        "FPCA explained variance ratios: %s",
        np.round(pca.explained_variance_ratio_, 3).tolist(),
    )

    check_data_quality(factors_df, "final FPCA factors")
    return factors_df

def main():
    """Main function to process option data and generate FPCA factors."""
    from iv_drl.utils import load_config
    CONFIG = load_config("data_config.yaml")
    
    input_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "iv_fpca_factors.parquet"
    
    if output_path.exists():
        logging.info('FPCA factors already exist -> %s', output_path)
        return
        
    if not input_dir.exists():
        logging.error("OptionMetrics parquet directory not found: %s", input_dir)
        return

    files = sorted(list(input_dir.glob("**/*.parquet")))
    if not files:
        logging.error("No parquet files found in %s or its subdirectories", input_dir)
        return

    logging.info("Found %d parquet files", len(files))
    
    try:
        all_data = pd.concat((pd.read_parquet(f) for f in tqdm(files, desc="Loading files")), ignore_index=True)
        logging.info(f"Total rows loaded from all files: {len(all_data)}")
        
        factors_df = calculate_fpca_factors(all_data)
        
        if factors_df.empty:
            logging.error("No FPCA factors generated.")
            return
            
        factors_df.to_parquet(output_path)
        logging.info('Saved FPCA factors -> %s', output_path)
        
    except Exception as e:
        logging.error("Error during FPCA calculation: %s", str(e))
        raise

if __name__ == "__main__":
    main() 