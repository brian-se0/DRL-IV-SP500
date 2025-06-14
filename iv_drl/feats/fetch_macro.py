"""
Module for fetching and processing macroeconomic data from FRED (Federal Reserve Economic Data).
This module handles data fetching, frequency conversion, and feature calculation for various economic indicators.
"""
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import time

import pandas as pd
import numpy as np
from fredapi import Fred
from scipy import interpolate
import yfinance as yf

# Core economic indicators from FRED with their descriptions
FRED_SERIES = {
    'GDP': 'Gross Domestic Product',
    'GDPC1': 'Real Gross Domestic Product',
    'GDPPOT': 'Real Potential Gross Domestic Product',
    'CPIAUCSL': 'Consumer Price Index for All Urban Consumers',
    'CPILFESL': 'Consumer Price Index for All Urban Consumers: All items less food and energy',
    'PCEPI': 'Personal Consumption Expenditures: Chain-type Price Index',
    'FEDFUNDS': 'Federal Funds Effective Rate',
    'DGS10': '10-Year Treasury Constant Maturity Rate',
    'DGS2': '2-Year Treasury Constant Maturity Rate',
    'DGS3MO': '3-Month Treasury Constant Maturity Rate',
    'PAYEMS': 'All Employees: Total Nonfarm Payrolls',
    'UNRATE': 'Unemployment Rate',
    'HOUST': 'Housing Starts: Total: New Privately Owned Housing Units Started',
    'CSUSHPISA': 'S&P/Case-Shiller U.S. National Home Price Index',
    'VIXCLS': 'CBOE Volatility Index: VIX',
    'M2': 'M2 Money Stock',
    'M2REAL': 'Real M2 Money Stock',
}

# Essential series that are always included in the analysis
DEFAULT_SERIES = ["VIXCLS", "DGS10", "DGS3MO"]

# Mapping between FRED frequency codes and API frequency parameters
FREQUENCY_MAP = {
    'D': 'd', 'W': 'w', 'BW': 'bw', 'M': 'm', 'Q': 'q', 'SA': 'sa', 'A': 'a',
    'W-SUN': 'w', 'W-MON': 'w', 'W-TUE': 'w', 'W-WED': 'w', 'W-THU': 'w', 'W-FRI': 'w', 'W-SAT': 'w'
}

# Mapping between FRED frequency codes and pandas frequency codes for resampling
PANDAS_FREQUENCY_MAP = {
    'd': 'D', 'w': 'W', 'bw': 'W', 'm': 'ME', 'q': 'QE', 'sa': '6M', 'a': 'Y',
    'w-sun': 'W-SUN', 'w-mon': 'W-MON', 'w-tue': 'W-TUE', 'w-wed': 'W-WED',
    'w-thu': 'W-THU', 'w-fri': 'W-FRI', 'w-sat': 'W-SAT'
}

def get_series_frequency(series_id: str) -> str:
    """
    Determine the natural frequency of a FRED series.
    Returns 'Q' for quarterly, 'M' for monthly, or 'D' for daily data.
    """
    quarterly_series = {'GDP', 'GDPC1', 'GDPPOT'}
    monthly_series = {
        'CPIAUCSL', 'CPILFESL', 'PCEPI', 'FEDFUNDS', 'PAYEMS', 'UNRATE',
        'HOUST', 'CSUSHPISA', 'M2', 'M2REAL'
    }
    daily_series = {'DGS10', 'DGS2', 'DGS3MO', 'VIXCLS'}
    
    if series_id in quarterly_series:
        return 'Q'
    elif series_id in monthly_series:
        return 'M'
    elif series_id in daily_series:
        return 'D'
    else:
        return 'M'

def aggregate_to_monthly(series, name):
    """
    Convert time series data to monthly frequency using appropriate aggregation methods.
    Handles different input frequencies (daily, weekly, quarterly) and preserves metadata.
    """
    if series.empty:
        return pd.Series(index=pd.date_range(start='2004-01-31', end='2021-04-30', freq='ME'))
    
    natural_freq = series.attrs.get('frequency_short', None)
    seasonal_adjustment = series.attrs.get('seasonal_adjustment', '')
    date_range = pd.date_range(start='2003-01-31', end='2021-04-30', freq='ME')
    
    # Handle different input frequencies
    if natural_freq == 'q':
        # For quarterly data, use cubic interpolation to monthly
        series = series.asfreq('QE')
        series = series.reindex(date_range)
        if not series.isna().all():
            try:
                series = series.interpolate(method='cubic')
            except:
                series = series.interpolate(method='linear')
    elif natural_freq == 'm':
        # For monthly data, ensure month-end alignment
        series = series.asfreq('ME')
        series = series.reindex(date_range)
    elif natural_freq == 'd':
        # For daily data, use last value for market data, mean for others
        if name in ['VIXCLS', 'DGS3MO', 'DGS2', 'DGS10']:
            series = series.resample('ME').last()
        else:
            series = series.resample('ME').mean()
        series = series.reindex(date_range)
    elif natural_freq == 'w':
        # For weekly data, use end-of-month values
        series = series.resample('ME').last()
        series = series.reindex(date_range)
    else:
        # For other frequencies, use monthly average
        series = series.resample('ME').mean()
        series = series.reindex(date_range)
    
    # Ensure month-end alignment and preserve metadata
    series.index = series.index.map(lambda x: x + pd.offsets.MonthEnd(0))
    series.index.freq = 'ME'
    series.attrs['frequency_short'] = natural_freq
    series.attrs['seasonal_adjustment'] = seasonal_adjustment
    series.attrs['aggregated_frequency'] = 'ME'
    
    return series

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset using appropriate methods for different types of data.
    Market prices are not filled, while economic indicators use limited interpolation.
    """
    imputed_df = df.copy()
    
    # Define series types for different imputation strategies
    market_prices = {'DGS10', 'DGS2', 'DGS3MO', 'VIXCLS', 'SP500'}
    economic_indicators = {'GDP', 'GDPC1', 'GDPPOT', 'CPIAUCSL', 'CPILFESL', 'PCEPI', 
                         'PAYEMS', 'UNRATE', 'HOUST', 'CSUSHPISA'}
    monetary_indicators = {'M2', 'M2REAL', 'FEDFUNDS'}
    
    for col in imputed_df.columns:
        if imputed_df[col].isnull().any():
            freq = get_series_frequency(col)
            
            # Apply different imputation strategies based on data type
            if col in market_prices:
                # Market prices should not be filled
                continue
            elif col in economic_indicators:
                # Economic indicators use limited interpolation
                if freq == 'Q':
                    imputed_df[col] = imputed_df[col].interpolate(method='cubic', limit=1)
                else:
                    imputed_df[col] = imputed_df[col].interpolate(method='linear', limit=1)
            elif col in monetary_indicators:
                # Monetary indicators allow slightly more filling
                if freq == 'M':
                    imputed_df[col] = imputed_df[col].interpolate(method='linear', limit=2)
                else:
                    imputed_df[col] = imputed_df[col].interpolate(method='linear', limit=1)
            
            # Forward fill derived metrics
            if any(x in col for x in ['_growth', '_volatility', '_change', '_return']):
                imputed_df[col] = imputed_df[col].ffill()
    
    return imputed_df

def calculate_macro_features(df: pd.DataFrame, options_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculate macroeconomic features from raw FRED data.
    Includes growth rates, spreads, and volatility measures.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Convert all series to monthly frequency
    features = df.copy()
    monthly_features = pd.DataFrame()
    
    for col in features.columns:
        if not features[col].empty:
            monthly_series = aggregate_to_monthly(features[col], col)
            monthly_features[col] = monthly_series
            monthly_features[col].index.freq = 'ME'
    
    features = monthly_features
    
    # Add S&P 500 data and calculate market metrics
    if options_data is not None and 'active_underlying_price_1545' in options_data.columns:
        # Calculate daily returns and volatility
        daily_returns = options_data['active_underlying_price_1545'].pct_change(fill_method=None)
        daily_volatility = daily_returns.rolling(window=21, min_periods=15).std() * np.sqrt(252)
        
        # Aggregate to monthly frequency
        sp500_monthly = options_data['active_underlying_price_1545'].resample('ME').last()
        returns_monthly = daily_returns.resample('ME').last()
        volatility_monthly = daily_volatility.resample('ME').last()
        
        features['SP500'] = sp500_monthly
        features['Market_return'] = returns_monthly
        features['Market_volatility'] = volatility_monthly
        
        features['SP500'].index.freq = 'ME'
        features['Market_return'].index.freq = 'ME'
        features['Market_volatility'].index.freq = 'ME'
    
    # Create complete date range and ensure consistent frequency
    date_range = pd.date_range(start=features.index.min(), end='2021-04-30', freq='ME')
    features = features.reindex(date_range)
    
    features.index.freq = 'ME'
    for col in features.columns:
        features[col].index.freq = 'ME'
    
    # Handle missing values
    features = impute_missing_values(features)
    
    # Calculate growth rates and spreads
    if 'GDP' in features.columns:
        features['GDP_growth'] = features['GDP'].pct_change(periods=3, fill_method=None) * 100
    
    if 'GDPC1' in features.columns:
        features['Real_GDP_growth'] = features['GDPC1'].pct_change(periods=3, fill_method=None) * 100
    
    if all(x in features.columns for x in ['GDPC1', 'GDPPOT']):
        features['Output_gap'] = (features['GDPC1'] - features['GDPPOT']) / features['GDPPOT']
    
    # Calculate inflation rates
    for col, name in [('CPIAUCSL', 'CPI'), ('CPILFESL', 'Core'), ('PCEPI', 'PCE')]:
        if col in features.columns:
            features[f'{name}_inflation'] = features[col].pct_change(periods=12, fill_method=None) * 100
    
    # Calculate interest rate spreads
    if all(x in features.columns for x in ['DGS10', 'DGS2']):
        features['Term_spread'] = features['DGS10'] - features['DGS2']
    
    if all(x in features.columns for x in ['DGS10', 'FEDFUNDS']):
        features['Policy_spread'] = features['DGS10'] - features['FEDFUNDS']
    
    # Calculate growth rates for employment and housing
    for col, name in [('PAYEMS', 'Employment'), ('HOUST', 'Housing_starts'), ('CSUSHPISA', 'Home_price')]:
        if col in features.columns:
            features[f'{name}_growth'] = features[col].pct_change(periods=12, fill_method=None) * 100
    
    # Calculate money supply growth
    for col, name in [('M2', 'M2'), ('M2REAL', 'Real_M2')]:
        if col in features.columns:
            features[f'{name}_growth'] = features[col].pct_change(periods=12, fill_method=None) * 100
    
    # Calculate VIX metrics
    if 'VIXCLS' in features.columns:
        features['VIX_change'] = features['VIXCLS'].pct_change(fill_method=None)
        features['VIX_volatility'] = features['VIXCLS'].rolling(window=21, min_periods=15).std()
    
    # Final imputation of any remaining missing values
    features = impute_missing_values(features)
    
    # Ensure consistent frequency across all series
    for col in features.columns:
        features[col].index.freq = 'ME'
    
    features.index.freq = 'ME'
    
    return features

class MacroDataFetcher:
    """
    Class for fetching and managing macroeconomic data from FRED.
    Handles rate limiting, retries, and data aggregation.
    """
    
    def __init__(self):
        """Initialize the fetcher using FRED API key from environment variable."""
        api_key = os.getenv('FRED_API_KEY')
        if not api_key:
            raise ValueError("FRED_API_KEY environment variable must be set")
        
        self.fred = Fred(api_key=api_key)
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests to stay under rate limit
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to stay under FRED API limits."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last_request)
        self.last_request_time = time.time()
    
    def get_series(self, 
                  series_id: str, 
                  start_date: Optional[Union[str, datetime]] = None,
                  end_date: Optional[Union[str, datetime]] = None,
                  max_retries: int = 3) -> pd.Series:
        """
        Fetch a single FRED series with retry logic and rate limiting.
        Handles different frequencies and preserves metadata.
        """
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                # Get series metadata and determine frequency
                try:
                    series_info = self.fred.get_series_info(series_id)
                    natural_freq = series_info.get('frequency_short', 'm')
                    seasonal_adjustment = series_info.get('seasonal_adjustment_short', '')
                    api_freq = FREQUENCY_MAP.get(natural_freq.upper(), 'm')
                except Exception as e:
                    api_freq = 'm'
                    natural_freq = 'm'
                    seasonal_adjustment = ''
                
                # Fetch the series data
                series_data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                    frequency=api_freq,
                    aggregation_method='avg'
                )
                
                if series_data is not None and not series_data.empty:
                    # Process and return the data
                    series_data = pd.Series(series_data, name=series_id)
                    series_data.index = pd.to_datetime(series_data.index)
                    
                    # Store metadata
                    series_data.attrs['frequency_short'] = natural_freq
                    series_data.attrs['seasonal_adjustment'] = seasonal_adjustment
                    series_data.attrs['api_frequency'] = api_freq
                    
                    # Ensure end-of-day values for daily data
                    if api_freq == 'd':
                        series_data = series_data.resample('D').last()
                    
                    return series_data
                
                return pd.Series()
                
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg:
                    # Rate limit hit, wait longer and retry
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                elif "400" in err_msg:
                    return pd.Series()
                elif "404" in err_msg:
                    return pd.Series()
                else:
                    if attempt == max_retries - 1:
                        return pd.Series()
                    time.sleep(1)
        
        return pd.Series()
    
    def get_multiple_series(self,
                          series_ids: Optional[List[str]] = None,
                          start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch multiple FRED series and combine them into a DataFrame.
        Includes default series if none are specified.
        """
        if series_ids is None:
            series_ids = list(FRED_SERIES.keys())
        else:
            series_ids = list(set(series_ids) | set(DEFAULT_SERIES))
        
        out = []
        for series_id in series_ids:
            series_data = self.get_series(
                series_id,
                start_date=start_date,
                end_date=end_date
            )
            if not series_data.empty:
                out.append(series_data)
        
        if not out:
            return pd.DataFrame()
            
        return pd.concat(out, axis=1)
    
    def get_all_series(self,
                      start_date: Optional[Union[str, datetime]] = None,
                      end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch all predefined FRED series."""
        return self.get_multiple_series(
            list(FRED_SERIES.keys()),
            start_date=start_date,
            end_date=end_date
        )

def get_spx_from_yfinance(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch S&P 500 data from yfinance for the specified date range.
    Returns data in the same format as the options dataset.
    """
    spx = yf.Ticker("^GSPC")
    df = spx.history(start=start_date, end=end_date)
    df = df.rename(columns={'Close': 'active_underlying_price_1545'})
    df = df[['active_underlying_price_1545']]
    df.index = df.index.tz_localize(None)
    return df

def load_spx_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load S&P 500 price data, combining yfinance data for 2003 with options dataset for 2004 onwards.
    Handles the transition between data sources and ensures consistent formatting.
    """
    transition_date = datetime(2004, 1, 2)
    combined_data = pd.DataFrame()
    
    # Get yfinance data for 2003
    if start_date < transition_date:
        yf_end = min(transition_date, end_date)
        yf_data = get_spx_from_yfinance(start_date, yf_end)
        combined_data = pd.concat([combined_data, yf_data])
    
    # Get options dataset data for 2004 onwards
    if end_date >= transition_date:
        options_start = max(transition_date, start_date)
        parquet_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'results', 'parquet_yearly')
        years = range(options_start.year, end_date.year + 1)
        
        dfs = []
        for year in years:
            year_dir = os.path.join(parquet_dir, str(year))
            if not os.path.exists(year_dir):
                continue
                
            parquet_files = [f for f in os.listdir(year_dir) if f.endswith('.parquet')]
            
            for file in parquet_files:
                file_path = os.path.join(year_dir, file)
                try:
                    df = pd.read_parquet(file_path, columns=['quote_date', 'active_underlying_price_1545'])
                    dfs.append(df)
                except Exception as e:
                    logging.error(f"Error reading {file_path}: {str(e)}")
        
        if dfs:
            options_data = pd.concat(dfs, ignore_index=True)
            options_data.set_index('quote_date', inplace=True)
            
            if options_data.index.tz is not None:
                options_data.index = options_data.index.tz_localize(None)
            
            options_data = options_data[
                (options_data.index >= options_start) & 
                (options_data.index <= end_date)
            ]
            combined_data = pd.concat([combined_data, options_data])
    
    # Ensure chronological order and handle duplicates
    combined_data = combined_data.sort_index()
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
    
    return combined_data

def main():
    """
    Main function to fetch and process macroeconomic data.
    Fetches FRED data, combines with S&P 500 data, and calculates features.
    """
    start_date = datetime(2003, 1, 2)
    end_date = datetime(2021, 4, 9, 23, 59, 59)
    
    # Fetch and process FRED data
    fetcher = MacroDataFetcher()
    raw_data = fetcher.get_all_series(start_date, end_date)
    
    # Convert to monthly frequency
    monthly_raw_data = pd.DataFrame()
    for col in raw_data.columns:
        if not raw_data[col].empty:
            monthly_series = aggregate_to_monthly(raw_data[col], col)
            monthly_raw_data[col] = monthly_series
    
    # Get S&P 500 data and calculate features
    spx_data = load_spx_data(start_date, end_date)
    features = calculate_macro_features(monthly_raw_data, spx_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_data_path = os.path.join('results', f'macro_raw_data_{timestamp}.csv')
    features_path = os.path.join('results', f'macro_features_{timestamp}.csv')
    
    monthly_raw_data.to_csv(raw_data_path)
    features.to_csv(features_path)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    main() 