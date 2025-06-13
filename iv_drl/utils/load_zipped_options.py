import argparse
import csv
from pathlib import Path
import sys
import pandas as pd
import zipfile
from iv_drl.utils import load_config
import logging
from datetime import datetime
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = load_config('data_config.yaml')
RAW_ZIP_DIR = Path(CONFIG['paths']['option_data_zip_dir']).resolve()
OUTPUT_DIR = Path(CONFIG['paths']['output_dir']).resolve() / 'parquet_yearly'

# Columns we actually need for later feature engineering
KEEP_COLS = {
    'quote_date': 'quote_date',
    'expiration': 'expiration',
    'option_type': 'call_put',
    'delta_1545': 'delta_1545',
    'implied_volatility_1545': 'implied_volatility_1545',
    'bid_1545': 'best_bid',
    'ask_1545': 'best_offer',
}

CHUNK_ROWS = 200_000  # adjust to fit memory comfortably


def process_zip(zip_path: Path) -> pd.DataFrame:
    """Extract SPX rows from a single ZIP (daily CSV) and return DataFrame."""
    with zipfile.ZipFile(zip_path) as zf:
        # assume only one file inside zip
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Use iterator reader to keep memory low
            iter_csv = pd.read_csv(
                f,
                usecols=list(KEEP_COLS.keys()) + ['underlying_symbol'],
                chunksize=CHUNK_ROWS,
            )
            parts = []
            for chunk in iter_csv:
                # Normalise symbol to remove leading '^' if present
                symbols = chunk['underlying_symbol'].str.upper().str.lstrip('^')
                spx_chunk = chunk[symbols == 'SPX']
                if not spx_chunk.empty:
                    parts.append(spx_chunk)
            if not parts:
                return pd.DataFrame(columns=KEEP_COLS.values())
            df = pd.concat(parts, ignore_index=True)
            df.rename(columns=KEEP_COLS, inplace=True)
            # Normalise option_type to 'C'/'P'
            df['call_put'] = df['call_put'].str.upper()
            return df


def main():
    parser = argparse.ArgumentParser(description='Convert OptionMetrics daily zip CSVs to yearly Parquet files for SPX only.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing yearly parquet files')
    args = parser.parse_args()

    if not RAW_ZIP_DIR.exists():
        sys.exit(f'Raw ZIP directory not found: {RAW_ZIP_DIR}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group zip files by year based on the date suffix
    yearly_batches = defaultdict(list)
    for zip_path in RAW_ZIP_DIR.glob('UnderlyingOptionsEODCalcs_*.zip'):
        date_str = zip_path.stem.split('_')[-1]  # 'YYYY-MM-DD'
        try:
            year = datetime.strptime(date_str, '%Y-%m-%d').year
            yearly_batches[year].append(zip_path)
        except ValueError:
            logging.warning(f'Skip unrecognised filename: {zip_path.name}')

    for year, files in sorted(yearly_batches.items()):
        out_path = OUTPUT_DIR / f'spx_{year}.parquet'
        if out_path.exists() and not args.overwrite:
            logging.info(f'Skip {year} (parquet already exists)')
            continue

        logging.info(f'Processing {year} with {len(files)} daily ZIPs...')
        year_frames = []
        for zip_path in sorted(files):
            df_day = process_zip(zip_path)
            if not df_day.empty:
                year_frames.append(df_day)
        if year_frames:
            year_df = pd.concat(year_frames, ignore_index=True)
            # Ensure dtypes
            year_df['quote_date'] = pd.to_datetime(year_df['quote_date'])
            year_df['expiration'] = pd.to_datetime(year_df['expiration'])
            # Save parquet partition
            year_df.to_parquet(out_path, index=False)
            logging.info(f'Written {len(year_df)} rows to {out_path}')
        else:
            logging.warning(f'No SPX rows found for {year}')

    logging.info('All done.')


if __name__ == '__main__':
    main() 