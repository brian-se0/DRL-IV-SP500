import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # Enable experimental feature
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Union, List
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

def kalman_impute(
    series: pd.Series,
    max_gap: int = 5,
    measurement_noise: float = 0.1,
    transition_noise: float = 0.1
) -> pd.Series:
    """
    Impute missing values using Kalman filter smoothing via SARIMAX.
    
    Parameters
    ----------
    series : pd.Series
        Time series with missing values
    max_gap : int, default=5
        Maximum gap size to impute
    measurement_noise : float, default=0.1
        Measurement noise parameter
    transition_noise : float, default=0.1
        Transition noise parameter
        
    Returns
    -------
    pd.Series
        Imputed series
    """
    if series.isna().sum() == 0:
        return series
        
    # Initialize SARIMAX model (AR(1) with measurement noise)
    model = SARIMAX(
        series,
        order=(1, 0, 0),  # AR(1)
        measurement_error=True,
        enforce_stationarity=False
    )
    
    # Fit model
    try:
        results = model.fit(disp=False)
        # Get smoothed state
        smoothed = results.get_prediction().predicted_mean
    except Exception as e:
        logger.warning(f"SARIMAX fitting failed: {str(e)}. Falling back to simple interpolation.")
        return series.interpolate(method='linear', limit=max_gap)
    
    # Create result series
    result = series.copy()
    result.iloc[series.isna()] = smoothed[series.isna()]
    
    # Only impute gaps up to max_gap
    mask = series.isna()
    for i in range(len(series)):
        if mask[i]:
            gap_size = 1
            j = i + 1
            while j < len(series) and mask[j]:
                gap_size += 1
                j += 1
            if gap_size > max_gap:
                result.iloc[i:j] = np.nan
                
    return result

def tdi_impute(
    df: pd.DataFrame,
    n_imputations: int = 5,
    max_iter: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Time-Dependent Iterative imputation using MICE with lagged predictors.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values
    n_imputations : int, default=5
        Number of imputations to perform
    max_iter : int, default=10
        Maximum number of iterations
    random_state : int, default=42
        Random state for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Imputed DataFrame
    """
    # Add lagged features
    df_lagged = df.copy()
    for col in df.columns:
        for lag in [1, 2, 3]:  # Add up to 3 lags
            df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    # Initialize imputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=100,
            random_state=random_state
        ),
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Impute
    imputed = imputer.fit_transform(df_lagged)
    
    # Convert back to DataFrame and keep only original columns
    result = pd.DataFrame(
        imputed,
        columns=df_lagged.columns,
        index=df.index
    )
    
    return result[df.columns]

def volatility_aware_impute(
    df: pd.DataFrame,
    window: int = 30,
    max_gap: int = 5
) -> pd.DataFrame:
    """
    Impute missing values with volatility awareness.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values
    window : int, default=30
        Window size for volatility calculation
    max_gap : int, default=5
        Maximum gap size to impute
        
    Returns
    -------
    pd.DataFrame
        Imputed DataFrame
    """
    result = df.copy()
    
    for col in df.columns:
        # Calculate rolling volatility
        vol = df[col].rolling(window=window).std()
        
        # Normalize by volatility
        normalized = df[col] / vol
        
        # Impute normalized series
        imputed = kalman_impute(normalized, max_gap=max_gap)
        
        # Scale back
        result[col] = imputed * vol
        
    return result 