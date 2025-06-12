import logging
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

from iv_drl.utils import load_config

warnings.filterwarnings(
    "ignore",
    message="The behavior of DatetimeProperties.to_pydatetime is deprecated.*",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
CONFIG = load_config("data_config.yaml")


def get_atm_iv(df_day: pd.DataFrame, ttm_days: int) -> float:
    mask = df_day["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    atm_calls = df_day.loc[mask & (df_day["call_put"] == "C")].copy()
    if atm_calls.empty:
        return np.nan
    atm_calls["delta_dist"] = (atm_calls["delta_1545"] - 0.5).abs()
    return atm_calls.loc[atm_calls["delta_dist"].idxmin()]["implied_volatility_1545"]


def get_skew(df_day: pd.DataFrame, ttm_days: int) -> float:
    mask = df_day["ttm_days"].between(ttm_days - 5, ttm_days + 5)
    calls = df_day.loc[mask & (df_day["call_put"] == "C")].copy()
    puts = df_day.loc[mask & (df_day["call_put"] == "P")].copy()
    if calls.empty or puts.empty:
        return np.nan
    calls["delta_dist"] = (calls["delta_1545"] - 0.25).abs()
    puts["delta_dist"] = (puts["delta_1545"] + 0.25).abs()
    iv_call = calls.loc[calls["delta_dist"].idxmin()]["implied_volatility_1545"]
    iv_put = puts.loc[puts["delta_dist"].idxmin()]["implied_volatility_1545"]
    return iv_call - iv_put


def calculate_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    logging.info("Pre-processing raw option data…")
    df["quote_date"] = pd.to_datetime(df["quote_date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["ttm_days"] = (df["expiration"] - df["quote_date"]).dt.days

    df["mid_price"] = (df["best_bid"] + df["best_offer"]) / 2
    df["bid_ask_spread_pct"] = (df["best_offer"] - df["best_bid"]) / df["mid_price"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    req = ["ttm_days", "delta_1545", "implied_volatility_1545", "bid_ask_spread_pct"]
    df.dropna(subset=req, inplace=True)
    df = df[df["ttm_days"] > 0]

    daily_rows = []
    for date, df_day in tqdm(df.groupby("quote_date"), desc="Daily IV features"):
        atm_30 = get_atm_iv(df_day, 30)
        atm_90 = get_atm_iv(df_day, 90)
        daily_rows.append(
            {
                "date": date,
                "atm_iv_30d": atm_30,
                "atm_iv_90d": atm_90,
                "term_structure_slope": atm_90 - atm_30 if pd.notna(atm_30) and pd.notna(atm_90) else np.nan,
                "skew_30d": get_skew(df_day, 30),
                "avg_bid_ask_spread_pct": df_day.loc[
                    df_day["ttm_days"].between(15, 45)
                    & df_day["delta_1545"].abs().between(0.2, 0.8),
                    "bid_ask_spread_pct",
                ].mean(),
            }
        )

    if not daily_rows:
        return pd.DataFrame()

    feats = pd.DataFrame(daily_rows).set_index("date")
    for lag in range(1, 6):
        feats[f"atm_iv_30d_lag_{lag}"] = feats["atm_iv_30d"].shift(lag)
    feats["atm_iv_30d_change_1d"] = feats["atm_iv_30d"].diff()
    return feats


def main():
    input_dir = Path(CONFIG["paths"]["output_dir"]).resolve() / "parquet_yearly"
    output_path = Path(CONFIG["paths"]["output_dir"]).resolve() / "iv_surface_daily_features.parquet"
    if not input_dir.exists():
        logging.error("OptionMetrics parquet directory not found: %s", input_dir)
        return

    files = list(input_dir.glob("spx_*.parquet"))
    if not files:
        logging.error("No parquet files in %s", input_dir)
        return

    logging.info("Loading %d parquet files…", len(files))
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    feat_df = calculate_surface_features(raw)
    if feat_df.empty:
        logging.warning("No IV-surface features generated.")
        return

    feat_df.dropna(subset=["atm_iv_30d_lag_5", "atm_iv_30d_change_1d"], inplace=True)
    feat_df.to_parquet(output_path)
    logging.info('Saved IV-surface features -> %s', output_path)


if __name__ == "__main__":
    main() 