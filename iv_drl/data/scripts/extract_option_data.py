import logging
import zipfile
from pathlib import Path
import pandas as pd
from tqdm import tqdm

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
            logging.info(f"File already exists: {output_path}")
            return False
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get the CSV file name from the zip
            csv_name = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            
            # Define columns we need for IV surface calculations
            needed_cols = [
                # Basic option information
                'underlying_symbol', 'quote_date', 'root', 'expiration', 'strike',
                'option_type',
                
                # Price and IV data
                'bid_1545', 'ask_1545', 'implied_volatility_1545',
                
                # Greeks
                'delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545',
                
                # Underlying price data
                'underlying_bid_1545', 'underlying_ask_1545',
                'implied_underlying_price_1545', 'active_underlying_price_1545',
                
                # Volume and liquidity
                'open_interest'
            ]
            
            # Process the CSV in chunks to save memory
            chunks = []
            chunk_size = 100000  # Adjust based on available memory
            
            with zip_ref.open(csv_name) as csv_file:
                # Read in chunks and filter for SPX options early in the process
                for chunk in pd.read_csv(
                    csv_file,
                    usecols=needed_cols,
                    chunksize=chunk_size,
                    low_memory=False,
                    parse_dates=['quote_date', 'expiration']
                ):
                    # Filter for SPX options early in the process
                    spx_chunk = chunk[chunk['underlying_symbol'] == '^SPX'].copy()
                    if not spx_chunk.empty:
                        # Convert numeric columns to appropriate types
                        numeric_cols = [
                            'strike', 'bid_1545', 'ask_1545', 'implied_volatility_1545',
                            'delta_1545', 'gamma_1545', 'theta_1545', 'vega_1545', 'rho_1545',
                            'underlying_bid_1545', 'underlying_ask_1545', 
                            'implied_underlying_price_1545', 'active_underlying_price_1545',
                            'open_interest'
                        ]
                        
                        for col in numeric_cols:
                            spx_chunk[col] = pd.to_numeric(spx_chunk[col], errors='coerce')
                        
                        chunks.append(spx_chunk)
            
            if not chunks:
                logging.warning(f"No SPX options found in {zip_path}")
                return False
            
            # Combine chunks and save as parquet
            df = pd.concat(chunks, ignore_index=True)
            
            # Add some basic validation
            logging.info(f"Processed {len(df)} SPX options records")
            logging.info(f"Date range: {df['quote_date'].min()} to {df['quote_date'].max()}")
            logging.info(f"Strike range: {df['strike'].min()} to {df['strike'].max()}")
            logging.info(f"Expiration range: {df['expiration'].min()} to {df['expiration'].max()}")
            
            df.to_parquet(output_path)
            return True
            
    except Exception as e:
        logging.error(f"Error processing {zip_path}: {str(e)}")
        return False

def main():
    # Input directory containing zip files
    input_dir = Path("D:/Single-Equity Option Prices 2004_2021")
    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return
    
    # Output directory for parquet files - changed to match build_iv_surface.py expectations
    output_dir = Path("results/parquet_yearly")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all zip files and sort by date
    zip_files = sorted(input_dir.glob("UnderlyingOptionsEODCalcs_*.zip"))
    
    if not zip_files:
        logging.error(f"No zip files found in {input_dir}")
        return
    
    logging.info(f"Found {len(zip_files)} zip files to process")
    logging.info(f"Date range: {zip_files[0].stem.split('_')[-1]} to {zip_files[-1].stem.split('_')[-1]}")
    
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