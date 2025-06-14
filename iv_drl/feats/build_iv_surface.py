import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from iv_drl.utils import load_config

warnings.filterwarnings(
    "ignore",
    message="The behavior of DatetimeProperties.to_pydatetime is deprecated.*",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CONFIG = load_config("data_config.yaml")


def get_atm_iv(df_day: pd.DataFrame, ttm_days: int) -> float:
    mask = df_day["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    atm_calls = df_day.loc[mask & (df_day["option_type"] == "C")].copy()
    if atm_calls.empty:
        return np.nan
    atm_calls["delta_dist"] = (atm_calls["delta_1545"] - 0.5).abs()
    return atm_calls.loc[atm_calls["delta_dist"].idxmin()]["implied_volatility_1545"]


def get_skew(df_day: pd.DataFrame, ttm_days: int) -> float:
    mask = df_day["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    calls = df_day.loc[mask & (df_day["option_type"] == "C")].copy()
    puts = df_day.loc[mask & (df_day["option_type"] == "P")].copy()
    if calls.empty or puts.empty:
        return np.nan
    calls["delta_dist"] = (calls["delta_1545"] - 0.25).abs()
    puts["delta_dist"] = (puts["delta_1545"] + 0.25).abs()
    iv_call = calls.loc[calls["delta_dist"].idxmin()]["implied_volatility_1545"]
    iv_put = puts.loc[puts["delta_dist"].idxmin()]["implied_volatility_1545"]
    return iv_call - iv_put


def calculate_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    logging.info("Pre-processing raw option data…")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ttm_days"] = (df["expiration"] - df["quote_date"]).dt.days

    # Use active underlying price for mid price
    df["mid_price"] = df["active_underlying_price_1545"]
    
    # Calculate bid-ask spread percentage using active underlying price
    df["bid_ask_spread_pct"] = (df["ask_1545"] - df["bid_1545"]) / df["active_underlying_price_1545"]
    
    # Use active underlying price
    df["underlying_mid"] = df["active_underlying_price_1545"]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter for valid data
    req = ["ttm_days", "delta_1545", "implied_volatility_1545", "bid_ask_spread_pct"]
    df.dropna(subset=req, inplace=True)
    df = df[df["ttm_days"] > 0]

    dates = sorted(df["quote_date"].unique())
    chunk_size = 20
    daily_rows = []
    
    for i in range(0, len(dates), chunk_size):
        chunk_dates = dates[i:i + chunk_size]
        logging.info(f"Processing dates {chunk_dates[0]} to {chunk_dates[-1]}")
        
        for date in tqdm(chunk_dates, desc="Daily IV features"):
            df_day = df[df["quote_date"] == date].copy()
            
            # Calculate IV features
            atm_30 = get_atm_iv(df_day, 30)
            atm_90 = get_atm_iv(df_day, 90)
            
            # Calculate additional features
            avg_underlying_price = df_day["underlying_mid"].mean()
            avg_open_interest = df_day["open_interest"].mean()
            
            # Calculate liquidity metrics
            liquid_mask = df_day["ttm_days"].between(15, 45) & df_day["delta_1545"].abs().between(0.2, 0.8)
            avg_spread = df_day.loc[liquid_mask, "bid_ask_spread_pct"].mean()
            avg_volume = df_day.loc[liquid_mask, "open_interest"].mean()
            
            daily_rows.append(
                {
                    "date": date,
                    "atm_iv_30d": atm_30,
                    "atm_iv_90d": atm_90,
                    "term_structure_slope": atm_90 - atm_30 if pd.notna(atm_30) and pd.notna(atm_90) else np.nan,
                    "skew_30d": get_skew(df_day, 30),
                    "avg_bid_ask_spread_pct": avg_spread,
                    "avg_open_interest": avg_open_interest,
                    "underlying_price": avg_underlying_price,
                    "liquid_options_volume": avg_volume
                }
            )
            del df_day

    if not daily_rows:
        return pd.DataFrame()

    feats = pd.DataFrame(daily_rows).set_index("date")
    
    # Add lagged features
    for lag in range(1, 6):
        feats[f"atm_iv_30d_lag_{lag}"] = feats["atm_iv_30d"].shift(lag)
    feats["atm_iv_30d_change_1d"] = feats["atm_iv_30d"].diff()
    
    # Add price changes
    feats["underlying_price_change_1d"] = feats["underlying_price"].pct_change()
    
    return feats


def main():
    input_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "iv_surface_daily_features.parquet"
    
    # Skip if output file already exists
    if output_path.exists():
        logging.info('IV surface features already exist -> %s', output_path)
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
                year_features = calculate_surface_features(year_data)
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
        year_features = calculate_surface_features(year_data)
        if not year_features.empty:
            all_features.append(year_features)
    
    if not all_features:
        logging.warning("No IV-surface features generated.")
        return
        
    # Combine all features
    feat_df = pd.concat(all_features)
    feat_df.dropna(subset=["atm_iv_30d_lag_5", "atm_iv_30d_change_1d"], inplace=True)
    feat_df.to_parquet(output_path)
    logging.info('Saved IV-surface features -> %s', output_path)


if __name__ == "__main__":
    main() 