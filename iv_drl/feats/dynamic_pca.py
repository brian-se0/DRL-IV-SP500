import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union, List, Dict
import logging

logger = logging.getLogger(__name__)

class DynamicPCA:
    """
    Dynamic Principal Component Analysis implementation.
    
    This implementation follows the approach described in:
    - "Generalised Dynamic PCA" (arXiv 2407.16024)
    - "Supervised Dynamic PCA" (JASA, in press 2024)
    """
    
    def __init__(
        self,
        n_components: int = 3,
        window: int = 252,
        lagged: bool = True,
        robust: Optional[str] = None
    ):
        """
        Initialize DynamicPCA.
        
        Parameters
        ----------
        n_components : int, default=3
            Number of components to extract
        window : int, default=252
            Rolling window size (default: 1 year of trading days)
        lagged : bool, default=True
            Whether to include lagged features
        robust : Optional[str], default=None
            If 'huber', use robust PCA variant
        """
        self.n_components = n_components
        self.window = window
        self.lagged = lagged
        self.robust = robust
        self.pca_models = {}
        self.scalers = {}
        
    def _prepare_features(
        self,
        X: pd.DataFrame,
        t: int
    ) -> np.ndarray:
        """Prepare features for time t, including lags if specified."""
        if t < self.window:
            return None
            
        # Get window data
        window_data = X.iloc[t-self.window:t]
        
        if self.lagged:
            # Add lagged features
            lagged_data = pd.DataFrame()
            for col in X.columns:
                for lag in [1, 2, 3]:  # Add up to 3 lags
                    lagged_data[f'{col}_lag_{lag}'] = X[col].shift(lag)
            window_data = pd.concat([window_data, lagged_data], axis=1)
            
        return window_data.dropna().values
        
    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fit and transform the data using dynamic PCA.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Transformed data with dynamic PCA factors
        """
        n_samples = len(X)
        result = pd.DataFrame(index=X.index)
        
        # Initialize arrays for factors
        for i in range(self.n_components):
            result[f'factor_{i+1}'] = np.nan
            
        # For each time point
        for t in range(self.window, n_samples):
            # Prepare features
            window_data = self._prepare_features(X, t)
            if window_data is None:
                continue
                
            # Scale data
            if t not in self.scalers:
                self.scalers[t] = StandardScaler()
                window_data_scaled = self.scalers[t].fit_transform(window_data)
            else:
                window_data_scaled = self.scalers[t].transform(window_data)
                
            # Fit PCA
            if t not in self.pca_models:
                self.pca_models[t] = PCA(
                    n_components=min(self.n_components, window_data.shape[1])
                )
                factors = self.pca_models[t].fit_transform(window_data_scaled)
            else:
                factors = self.pca_models[t].transform(window_data_scaled)
                
            # Store latest factor values
            for i in range(self.n_components):
                if i < factors.shape[1]:
                    result.iloc[t, i] = factors[-1, i]
                    
        return result
        
    def transform(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform new data using fitted dynamic PCA models.
        
        Parameters
        ----------
        X : pd.DataFrame
            New data to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed data
        """
        return self.fit_transform(X)  # In dynamic PCA, we always refit 