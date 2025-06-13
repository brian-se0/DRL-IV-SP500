from pathlib import Path
import logging
import numpy as np
import pandas as pd
import yfinance as yf

from iv_drl.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config('data_config.yaml')


def fetch_spx_data(start_date='2000-01-01', end_date=pd.Timestamp.today().strftime('%Y-%m-%d')):
    """Fetch SPX OHLCV data from Yahoo Finance."""
    logging.info("Fetching SPX data from Yahoo Finance…")
    spx = yf.Ticker("^GSPC")
    hist = spx.history(start=start_date, end=end_date)
    if hist.empty:
        logging.error("No SPX data returned. Check ticker/date range.")
        return None
    logging.info("Fetched %d rows of SPX data.", len(hist))
    hist.index = pd.to_datetime(hist.index.date)
    return hist


def calculate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive realised-moment and microstructure features from price data."""
    if df is None:
        return pd.DataFrame()

    logging.info("Calculating price features…")
    features = df.copy()
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

    # Amihud Illiquidity
    dollar_volume = features['Close'] * features['Volume']
    features['amihud_illiq'] = np.abs(features['log_return']) / dollar_volume * 1e6

    for win in (5, 21):
        features[f'garman_klass_rv_{win}d'] = features['garman_klass_rv'].rolling(win).mean()
        features[f'parkinson_rv_{win}d'] = features['parkinson_rv'].rolling(win).mean()
        features[f'amihud_illiq_{win}d_mean'] = features['amihud_illiq'].rolling(win).mean()

    logging.info("Finished price feature calc.")
    return features.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'])


def main():
    output_dir = Path(CONFIG['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'spx_daily_features.parquet'

    # Skip if output file already exists
    if output_path.exists():
        logging.info('Price features already exist -> %s', output_path)
        return

    start_date = CONFIG['settings']['start_date']
    end_date = CONFIG['settings']['end_date']

    spx_data = fetch_spx_data(start_date, end_date)
    if spx_data is None:
        return

    feat_df = calculate_price_features(spx_data)
    feat_df.dropna(inplace=True)
    feat_df.to_parquet(output_path)
    logging.info('Saved price features -> %s', output_path)


if __name__ == "__main__":
    main() 