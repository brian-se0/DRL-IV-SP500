import pandas as pd
from econ499.feats.build_price import calculate_price_features


def test_calculate_price_features():
    df = pd.DataFrame({
        "Open": [1.0, 1.1, 1.2],
        "High": [1.2, 1.2, 1.3],
        "Low": [0.9, 1.0, 1.1],
        "Close": [1.1, 1.2, 1.25],
        "Dividends": [0, 0, 0],
        "Stock Splits": [0, 0, 0],
        "spy_close": [100, 101, 102],
        "spy_volume": [1e6, 2e6, 1.5e6],
    }, index=pd.date_range("2020-01-01", periods=3))
    feats = calculate_price_features(df)
    assert "amihud_illiq" in feats.columns
    assert len(feats) == 3
