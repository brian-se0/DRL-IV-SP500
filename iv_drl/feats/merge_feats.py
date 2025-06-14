import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import glob
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import stats
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.robust.scale import huber
from iv_drl.feats.imputation import kalman_impute, tdi_impute, volatility_aware_impute
from iv_drl.feats.dynamic_pca import DynamicPCA
from statsmodels.robust import Huber

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

def validate_data(df: pd.DataFrame, name: str) -> bool:
    """Validate input dataframe."""
    logger.debug(f"Validating {name} dataframe")
    if df.empty:
        logger.error(f"{name} dataframe is empty")
        return False
    if df.index.isnull().any():
        logger.error(f"{name} dataframe has null values in index")
        return False
    return True

def safe_imputation(series: pd.Series, window: int = 5) -> pd.Series:
    """Safely impute missing values in a series."""
    result = series.copy()
    result = result.fillna(method='ffill', limit=window)
    result = result.fillna(method='bfill', limit=window)
    return result

def check_data_quality(df: pd.DataFrame, stage: str) -> None:
    """Check data quality at different stages."""
    logger.debug(f"Data quality check at {stage}")
    logger.debug(f"Shape: {df.shape}")
    logger.debug(f"Missing values: {df.isna().sum().sum()}")
    logger.debug(f"Columns with missing values: {df.columns[df.isna().any()].tolist()}")

def resample_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Resample macro features to business days."""
    logger.debug("Resampling macro features")
    result = df.resample('B').ffill()
    logger.debug(f"Resampled shape: {result.shape}")
    return result

def multiple_imputation(df: pd.DataFrame, feature_type: str, n_imputations: int = 5) -> pd.DataFrame:
    """Perform multiple imputation on missing values."""
    logger.debug(f"Multiple imputation for {feature_type}")
    logger.debug(f"Initial missing values: {df.isna().sum().sum()}")
    
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=42),
        max_iter=10,
        random_state=42
    )
    
    result = pd.DataFrame(
        imputer.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    
    logger.debug(f"Final missing values: {result.isna().sum().sum()}")
    return result

def winsorize_huber(series: pd.Series, c: float = 1.345) -> pd.Series:
    """Apply Huber M-estimator to handle outliers."""
    # Skip non-numeric data
    if not np.issubdtype(series.dtype, np.number):
        return series
        
    # Skip if all values are the same
    if series.nunique() <= 1:
        return series
        
    # Create Huber instance with more iterations and looser tolerance
    huber_instance = Huber(
        c=c,
        tol=1e-6,  # Looser tolerance
        maxiter=100  # More iterations
    )
    
    try:
        # Calculate initial values
        initial_mu = np.median(series.values)
        initial_scale = np.median(np.abs(series.values - initial_mu)) * 1.4826  # MAD scale estimate
        
        # Calculate Huber M-estimator with initial values
        mu, scale = huber_instance(
            series.values,
            mu=initial_mu,
            initscale=initial_scale
        )
        
        # Calculate residuals and weights
        resid = series - mu
        wt = np.where(np.abs(resid) <= c*scale, 1, c*scale/np.abs(resid))
        
        # Apply winsorization
        return pd.Series(mu + resid*wt, index=series.index)
        
    except ValueError as e:
        logger.warning(f"Huber estimation failed for column {series.name}: {str(e)}. Using median-based winsorization.")
        # Fallback to simple median-based winsorization
        median = series.median()
        mad = np.median(np.abs(series - median))
        upper = median + c * mad
        lower = median - c * mad
        return series.clip(lower=lower, upper=upper)

def handle_missing_values(df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    """Handle missing values in the dataframe using advanced methods."""
    logger.debug(f"Handling missing values for {feature_type}")
    logger.debug(f"Initial missing values: {df.isna().sum().sum()}")
    
    result = df.copy()
    
    # First apply robust outlier handling only to numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        try:
            result[col] = winsorize_huber(result[col])
        except Exception as e:
            logger.warning(f"Failed to winsorize column {col}: {str(e)}. Skipping winsorization.")
    
    # Handle missing values based on feature type
    if feature_type == 'price':
        # Use Kalman imputation for price data
        for col in result.columns:
            result[col] = kalman_impute(result[col], max_gap=5)
    elif feature_type == 'iv':
        # Use volatility-aware imputation for IV data
        result = volatility_aware_impute(result, window=30, max_gap=5)
    elif feature_type == 'macro':
        # Use time-dependent iterative imputation for macro data
        result = tdi_impute(result)
    
    logger.debug(f"Final missing values: {result.isna().sum().sum()}")
    return result

def analyze_date_overlap(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> None:
    """Analyze date overlap between two dataframes."""
    logger.debug(f"Date overlap between {name1} and {name2}")
    common_dates = df1.index.intersection(df2.index)
    logger.debug(f"Common dates: {len(common_dates)}")

def handle_critical_iv_features(df: pd.DataFrame, critical_iv_cols: List[str]) -> pd.DataFrame:
    """Handle critical IV features."""
    logger.debug("Handling critical IV features")
    missing_rows = df[critical_iv_cols].isna().any(axis=1)
    logger.debug(f"Rows with missing critical IV: {missing_rows.sum()}")
    return df

def analyze_missing_patterns(df: pd.DataFrame) -> None:
    """Analyze patterns in missing data."""
    logger.debug("Analyzing missing patterns")
    missing_rates = df.isna().mean() * 100
    logger.debug(f"Missing rates > 5%: {missing_rates[missing_rates > 5].to_dict()}")

def analyze_date_ranges(dfs: Dict[str, pd.DataFrame]) -> None:
    """Analyze date ranges across dataframes."""
    logger.debug("Analyzing date ranges")
    for name, df in dfs.items():
        logger.debug(f"{name}: {df.index.min()} to {df.index.max()}")

def analyze_missing_patterns_detailed(df: pd.DataFrame) -> None:
    """Perform detailed analysis of missing data patterns."""
    logger.debug("Detailed missing pattern analysis")
    missing_rates = df.isna().mean() * 100
    high_missing = missing_rates[missing_rates > 5]
    logger.debug(f"Features with >5% missing: {high_missing.to_dict()}")

def analyze_outliers_detailed(df: pd.DataFrame) -> None:
    """Perform detailed analysis of outliers."""
    logger.debug("Analyzing outliers")
    for col in df.select_dtypes(include=[np.number]).columns:
        stats = df[col].describe()
        Q1 = stats['25%']
        Q3 = stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if len(outliers) > 0:
            logger.debug(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

def robust_scale_features(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply robust scaling to features."""
    logger.debug("Applying robust scaling")
    df_scaled = df.copy()
    
    for group_name, features in feature_groups.items():
        if not all(f in df.columns for f in features):
            logger.warning(f"Missing features in group {group_name}")
            continue
            
        scaler = RobustScaler()
        df_scaled[features] = scaler.fit_transform(df[features])
        
    return df_scaled

def apply_factor_analysis(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    n_components: int = 3,
    window: int = 252
) -> pd.DataFrame:
    """
    Apply dynamic factor analysis to reduce dimensionality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    feature_groups : Dict[str, List[str]]
        Dictionary mapping group names to feature lists
    n_components : int, default=3
        Number of components to extract
    window : int, default=252
        Rolling window size for dynamic PCA
        
    Returns
    -------
    pd.DataFrame
        Data with dynamic PCA factors
    """
    logger.debug("Applying dynamic factor analysis")
    df_factors = df.copy()
    
    for group_name, features in feature_groups.items():
        if not all(f in df.columns for f in features):
            logger.warning(f"Missing features in group {group_name}")
            continue
            
        group_data = df[features].copy()
        
        if group_data.isna().any().any():
            logger.debug(f"Imputing missing values in {group_name}")
            group_data = volatility_aware_impute(group_data)
        
        # Apply dynamic PCA
        dpca = DynamicPCA(
            n_components=n_components,
            window=window,
            lagged=True,
            robust='huber'
        )
        
        factors = dpca.fit_transform(group_data)
        
        # Add factors to result
        for i in range(n_components):
            df_factors[f'{group_name}_factor_{i+1}'] = factors[f'factor_{i+1}']
            
        # Log explained variance
        if hasattr(dpca.pca_models[list(dpca.pca_models.keys())[-1]], 'explained_variance_ratio_'):
            explained_variance = dpca.pca_models[list(dpca.pca_models.keys())[-1]].explained_variance_ratio_
            logger.debug(f"{group_name} explained variance: {explained_variance}")
        
    return df_factors

def main():
    """Main function to merge and process features."""
    try:
        logger.debug("Starting feature merging process")
        
        # Set paths
        output_dir = Path('results')
        output_path = output_dir / 'spx_iv_drl_state.csv'
        
        # Load feature files
        required_files = {
            'price': 'spx_daily_features.parquet',
            'iv': 'iv_surface_daily_features.parquet',
            'fpca': 'iv_fpca_factors.parquet'
        }
        
        # Check and load files
        for file_type, filename in required_files.items():
            file_path = output_dir / filename
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load features
        price_features = pd.read_parquet(output_dir / required_files['price'])
        iv_features = pd.read_parquet(output_dir / required_files['iv'])
        fpca_features = pd.read_parquet(output_dir / required_files['fpca'])
        
        logger.debug("Loaded all feature files")
        
        # Validate input data
        for name, df in [('price', price_features), ('iv', iv_features), ('fpca', fpca_features)]:
            if not validate_data(df, name):
                raise ValueError(f"Invalid {name} data")
        
        # Calculate common date range
        hf_start_date = max(price_features.index.min(), iv_features.index.min())
        hf_end_date = min(price_features.index.max(), iv_features.index.max())
        
        logger.debug(f"Common date range: {hf_start_date} to {hf_end_date}")
        
        # Align features to business days
        price_features = price_features.loc[hf_start_date:hf_end_date]
        iv_features = iv_features.loc[hf_start_date:hf_end_date]
        fpca_features = fpca_features.loc[hf_start_date:hf_end_date]
        
        # Handle missing values
        price_features = handle_missing_values(price_features, 'price')
        iv_features = handle_missing_values(iv_features, 'iv')
        fpca_features = handle_missing_values(fpca_features, 'fpca')
        
        # Merge features
        df = pd.merge(price_features, iv_features, left_index=True, right_index=True)
        df = pd.merge(df, fpca_features, left_index=True, right_index=True)
        
        logger.debug(f"Final merged shape: {df.shape}")
        
        # Save combined features
        df.to_csv(output_path)
        logger.info(f"Saved combined features -> {output_path}")
        
        logger.debug("Feature merging completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
