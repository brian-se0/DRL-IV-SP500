# noqa: D
import logging
from datetime import datetime
import os
from pathlib import Path

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import numpy as np

from iv_drl.utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CONFIG = load_config("data_config.yaml")

ADDITIONAL_FRED_SERIES = {
    "TEDRATE": "TEDRATE",
    "BAA": "BAA",
    "AAA": "AAA",
    "STLFSI2": "STLFSI2",
    "USEPUINDXD": "USEPUINDXD",
}

YF_EXTRA_TICKERS = {
    "VIX3M": "^VIX3M",
    "MOVE": "^MOVE",   # BofA MOVE index via Yahoo Finance
}

VVIX_CSV_PATH = Path("data_processed/VVIX_History.csv").resolve()


def fetch_fred_data(api_key=None, start_date="2000-01-01", end_date=datetime.today(), series=None):
    default_series = ["VIXCLS", "DGS10", "DGS3MO"]
    fred_series = default_series + list(ADDITIONAL_FRED_SERIES.values()) if series is None else list(set(series) | set(default_series))
    logging.info("Fetching FRED series individually: %s", fred_series)

    if api_key:
        os.environ["FRED_API_KEY"] = str(api_key)

    out = []
    for sid in fred_series:
        try:
            s = web.DataReader(sid, "fred", start_date, end_date)[sid]
            s.name = sid
            out.append(s)
            continue  # successfully fetched via FRED
        except Exception as e:
            err_msg = str(e).split("\n", 1)[0][:200]
            logging.warning("FRED series %s failed (%s) – trying fallback if available.", sid, err_msg)

    return pd.concat(out, axis=1) if out else pd.DataFrame()


def fetch_vvix(start_date="2000-01-01", end_date=datetime.today().strftime("%Y-%m-%d")):
    """Fetch VVIX data from local CSV file.
    
    The VVIX data is sourced from CBOE's historical data page:
    https://www.cboe.com/tradable_products/vix/vix_historical_data/
    """
    if not VVIX_CSV_PATH.exists():
        logging.error("VVIX CSV file not found at %s", VVIX_CSV_PATH)
        return pd.DataFrame()
        
    logging.info("Loading VVIX from CSV …")
    df = pd.read_csv(VVIX_CSV_PATH, parse_dates=["DATE"]).rename(columns={"DATE": "date"}).set_index("date")
    return df.loc[start_date:end_date]


def fetch_yf_series(tickers: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    out = []
    for name, tick in tickers.items():
        try:
            hist = yf.Ticker(tick).history(start=start_date, end=end_date)
            if hist.empty:
                continue
            s = hist["Close"].copy()
            s.index = pd.to_datetime(s.index.date)
            s.name = name
            out.append(s)
        except Exception as e:
            logging.warning("yfinance error for %s: %s", tick, e)
    return pd.concat(out, axis=1) if out else pd.DataFrame()


def calculate_macro_features(fred_df, vvix_df, extra_yf_df):
    if fred_df.empty and vvix_df.empty and extra_yf_df.empty:
        return pd.DataFrame()

    # Ensure all DataFrames have timezone-naive indices
    for df in [fred_df, vvix_df, extra_yf_df]:
        if not df.empty:
            df.index = df.index.tz_localize(None)

    df = pd.concat([fred_df, vvix_df, extra_yf_df], axis=1)
    df.index.name = "date"
    df.ffill(inplace=True)

    if "VIXCLS" in df.columns:
        df["VIX_daily_change"] = df["VIXCLS"].diff()

    if {"DGS10", "DGS3MO"}.issubset(df.columns):
        df["term_spread_10y_3m"] = pd.to_numeric(df["DGS10"], errors="coerce") - pd.to_numeric(df["DGS3MO"], errors="coerce")

    if {"BAA", "AAA"}.issubset(df.columns):
        df["credit_spread_baa_aaa"] = pd.to_numeric(df["BAA"], errors="coerce") - pd.to_numeric(df["AAA"], errors="coerce")

    if {"VIX3M", "VIXCLS"}.issubset(df.columns):
        df["vix_ts_slope"] = df["VIX3M"] - df["VIXCLS"]

    if {"VIXCLS", "MOVE"}.issubset(df.columns):
        df["VIX_MOVE_ratio"] = df["VIXCLS"] / df["MOVE"]

    return df


def main():
    out_path = Path(CONFIG["paths"]["output_dir"]) / "macro_daily_features.parquet"
    
    # Skip if output file already exists
    if out_path.exists():
        logging.info('Macro features already exist -> %s', out_path)
        return
        
    start_date = CONFIG["settings"]["start_date"]
    end_date = CONFIG["settings"]["end_date"]
    api_key = CONFIG["settings"]["fred_api_key"]

    fred = fetch_fred_data(api_key, start_date, end_date)
    vvix = fetch_vvix(start_date, end_date)
    extras = fetch_yf_series(YF_EXTRA_TICKERS, start_date, end_date)

    feat = calculate_macro_features(fred, vvix, extras)
    if feat.empty:
        logging.warning("No macro features generated.")
        return

    feat.dropna(how="all", inplace=True)
    feat.to_parquet(out_path)
    logging.info('Saved macro features -> %s', out_path)


if __name__ == "__main__":
    main() 