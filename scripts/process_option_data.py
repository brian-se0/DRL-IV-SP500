import logging
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from econ499.utils import load_config

CONFIG = load_config("data_config.yaml")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_zip_file(zip_path: Path, output_dir: Path) -> bool:
    """Process a single zip file containing OptionMetrics data.
    
    Returns:
        bool: True if file was processed, False if it was skipped
    """
    try:
        # Extract date from filename (e.g., UnderlyingOptionsEODCalcs_2004-01-02.zip)
        date_str = zip_path.stem.split('_')[-1]
        date = pd.to_datetime(date_str)
        
        # Create year directory if it doesn't exist
        year_dir = output_dir / str(date.year)
        year_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if output file already exists
        output_path = year_dir / f"{date_str}.parquet"
        if output_path.exists():
            return False
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the CSV file name from the zip
            csv_name = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            
            # Define columns we need - updated to match actual data format
            needed_cols = [
                'underlying_symbol', 'quote_date', 'root', 'expiration', 'strike',
                'option_type', 'bid_1545', 'ask_1545', 'implied_volatility_1545',
                'delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545',
                'underlying_bid_1545', 'underlying_ask_1545', 'implied_underlying_price_1545',
                'active_underlying_price_1545', 'open_interest'
            ]
            
            # Read only needed columns and filter for SPX
            with zip_ref.open(csv_name) as csv_file:
                df = pd.read_csv(
                    csv_file,
                    usecols=needed_cols,
                    low_memory=False
                )
            
            # Filter for SPX options
            df = df[df['underlying_symbol'] == '^SPX']  # Note the ^ prefix
            
            if df.empty:
                logging.warning(f"No SPX options found in {zip_path}")
                return False
            
            # Save as parquet
            df.to_parquet(output_path)
            return True
            
    except Exception as e:
        logging.error(f"Error processing {zip_path}: {str(e)}")
        return False

def main():
    # Input directory containing zip files
    input_dir = Path(CONFIG["paths"]["option_data_zip_dir"]).resolve()

    # Output directory for parquet files
    output_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all zip files
    zip_files = sorted(input_dir.glob("UnderlyingOptionsEODCalcs_*.zip"))
    
    if not zip_files:
        logging.error(f"No zip files found in {input_dir}")
        return
    
    logging.info(f"Found {len(zip_files)} zip files to process")
    
    # Process each zip file with progress bar
    processed = 0
    skipped = 0
    for zip_path in tqdm(zip_files, desc="Processing zip files"):
        if process_zip_file(zip_path, output_dir):
            processed += 1
        else:
            skipped += 1
    
    logging.info(f"Finished processing: {processed} files processed, {skipped} files skipped")

if __name__ == "__main__":
    main() 