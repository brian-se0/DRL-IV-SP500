from pathlib import Path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf

from iv_drl.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config('data_config.yaml')

def fetch_spx_volume(start_date='2004-01-02', end_date='2021-04-09'):
    """Fetch SPX trading volume from yfinance for the specified date range."""
    logging.info("Fetching SPX volume from yfinance...")
    spx = yf.download('^GSPC', start=start_date, end=end_date)
    if spx.empty:
        logging.warning("No SPX data found from yfinance.")
        return pd.DataFrame()
    
    # Flatten the Volume data and create DataFrame
    volume_series = spx['Volume'].squeeze()  # Convert to 1D Series
    volume_df = pd.DataFrame({
        'quote_date': pd.to_datetime(spx.index),
        'Volume': volume_series.values  # Use values from squeezed Series
    })
    
    logging.info("SPX volume fetched successfully.")
    return volume_df

def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive realised-moment and microstructure features from price data."""
    if df.empty:
        return pd.DataFrame()

    logging.info("Calculating price features…")
    
    # Debug: Print available columns
    logging.info("Available columns: %s", df.columns.tolist())
    
    # Create daily price data from options data
    daily_data = df.groupby('quote_date').agg({
        'active_underlying_price_1545': 'last',  # Close price
        'underlying_bid_1545': 'min',  # Low price
        'underlying_ask_1545': 'max',  # High price
    }).rename(columns={
        'active_underlying_price_1545': 'Close',
        'underlying_bid_1545': 'Low',
        'underlying_ask_1545': 'High',
    })
    
    # Add Open price (use previous day's close)
    daily_data['Open'] = daily_data['Close'].shift(1)
    
    features = daily_data.copy()
    features.index.name = 'date'

    # Daily log return
    features['log_return'] = np.log(features['Close'] / features['Close'].shift(1))

    WINDOW = 22  # ≈ one trading month

    # Realised variance (annualised)
    features['rv_22d'] = (features['log_return'] ** 2).rolling(WINDOW).sum() * (252 / WINDOW)

    # Realised skewness & kurtosis
    def _rskew(arr):
        num = np.nansum(arr ** 3)
        den = np.nansum(arr ** 2) ** 1.5
        return np.sqrt(len(arr)) * num / den if den else np.nan

    def _rkurt(arr):
        num = np.nansum(arr ** 4)
        den = np.nansum(arr ** 2) ** 2
        return len(arr) * num / den if den else np.nan

    features['realised_skew_22d'] = features['log_return'].rolling(WINDOW, min_periods=WINDOW).apply(_rskew, raw=True)
    features['realised_kurt_22d'] = features['log_return'].rolling(WINDOW, min_periods=WINDOW).apply(_rkurt, raw=True)

    # Realised quarticity
    features['realised_quarticity_22d'] = (WINDOW / 3.0) * (features['log_return'] ** 4).rolling(WINDOW).sum()

    # Jump flag (simple Lee–Mykland proxy)
    rolling_std = features['log_return'].rolling(WINDOW).std()
    features['jump_flag'] = (features['log_return'].abs() > 4 * rolling_std).astype(int)

    # Garman–Klass & Parkinson RV
    features['garman_klass_rv'] = 0.5 * np.square(np.log(features['High'] / features['Low'])) - (2 * np.log(2) - 1) * np.square(np.log(features['Close'] / features['Open']))
    features['parkinson_rv'] = (1 / (4 * np.log(2))) * np.square(np.log(features['High'] / features['Low']))

    # Fetch SPX volume from yfinance
    spx_volume = fetch_spx_volume()
    if not spx_volume.empty:
        # Debug logging
        logging.info("Features DataFrame structure before merge:")
        logging.info("Features index levels: %s", features.index.names)
        logging.info("Features columns: %s", features.columns.tolist())
        
        logging.info("SPX volume DataFrame structure before merge:")
        logging.info("SPX volume index levels: %s", spx_volume.index.names)
        logging.info("SPX volume columns: %s", spx_volume.columns.tolist())
        
        # Reset index of features and ensure date columns are datetime
        features = features.reset_index()
        features['quote_date'] = pd.to_datetime(features['date'])
        
        # Ensure SPX volume has correct datetime format and is flat
        spx_volume = spx_volume.reset_index()  # Reset any multi-level index
        spx_volume['quote_date'] = pd.to_datetime(spx_volume['quote_date'])
        
        # Debug logging after preparation
        logging.info("Features shape before merge: %s", features.shape)
        logging.info("SPX volume shape before merge: %s", spx_volume.shape)
        
        # Merge on quote_date
        features = features.merge(spx_volume, on='quote_date', how='left')
        
        # Set the index back to date
        features.set_index('date', inplace=True)
        logging.info("SPX volume merged into features.")
    else:
        logging.warning("SPX volume not available from yfinance.")

    # Amihud Illiquidity
    dollar_volume = features['Close'] * features['Volume']
    features['amihud_illiq'] = np.abs(features['log_return']) / dollar_volume * 1e6

    for win in (5, 21):
        features[f'garman_klass_rv_{win}d'] = features['garman_klass_rv'].rolling(win).mean()
        features[f'parkinson_rv_{win}d'] = features['parkinson_rv'].rolling(win).mean()
        features[f'amihud_illiq_{win}d_mean'] = features['amihud_illiq'].rolling(win).mean()

    logging.info("Finished price feature calc.")
    return features.drop(columns=['Open', 'High', 'Low', 'Volume'])

def main():
    input_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "spx_daily_features.parquet"
    
    # Skip if output file already exists
    if output_path.exists():
        logging.info('Price features already exist -> %s', output_path)
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
    
    # Process all files together
    logging.info("Processing all files together")
    all_data = pd.concat((pd.read_parquet(f) for f in tqdm(files, desc="Loading files")), ignore_index=True)
    
    feat_df = calculate_price_features(all_data)
    if feat_df.empty:
        logging.warning("No price features generated.")
        return
        
    feat_df.dropna(inplace=True)
    feat_df.to_parquet(output_path)
    logging.info('Saved price features -> %s', output_path)

if __name__ == "__main__":
    main() 