import pandas as pd
import zipfile
from pathlib import Path

from econ499.utils.load_zipped_options import process_zip


def test_process_zip_extracts_spx_rows(tmp_path):
    df = pd.DataFrame(
        {
            "quote_date": ["2021-01-01"] * 4,
            "expiration": ["2021-01-02"] * 4,
            "option_type": ["call", "put", "call", "put"],
            "delta_1545": [0.1, -0.1, 0.2, -0.2],
            "implied_volatility_1545": [0.1, 0.2, 0.3, 0.4],
            "bid_1545": [1, 2, 3, 4],
            "ask_1545": [1.1, 2.2, 3.3, 4.4],
            "underlying_symbol": ["SPX", "^SPX", "SPY", "SPXW"],
        }
    )
    csv_bytes = df.to_csv(index=False).encode()
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data.csv", csv_bytes)

    out = process_zip(zip_path)
    assert len(out) == 2
    expected_cols = {
        "quote_date",
        "expiration",
        "call_put",
        "delta_1545",
        "implied_volatility_1545",
        "best_bid",
        "best_offer",
        "underlying_symbol",
    }
    assert set(out.columns) == expected_cols
    assert set(out["call_put"]) == {"CALL", "PUT"}
