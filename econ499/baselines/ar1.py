from __future__ import annotations

"""Simple AR(1) baseline for ATM-IV forecasting."""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm

from econ499.utils import load_config

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()


def run_ar1(*, out_csv: str | Path | None = None, train_ratio: float = 0.8) -> Path:
    """Fit AR(1) on the ATM-IV series and save out-of-sample forecasts."""
    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    iv = df[CFG["features"]["target_col"]].astype(float)

    df["IV_next"] = iv.shift(-1)
    df.dropna(inplace=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    oos_df = df.iloc[split_idx:]

    X_train = sm.add_constant(train_df[CFG["features"]["target_col"]])
    model = sm.OLS(train_df["IV_next"], X_train).fit()

    X_all = sm.add_constant(df[CFG["features"]["target_col"]])
    df["ar1_forecast"] = model.predict(X_all)

    out_path = Path(out_csv) if out_csv else OUT_DIR / "ar1_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[oos_df.index, ["date", "ar1_forecast"]].to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print("Saved AR(1) forecasts to", out_path)
    return out_path


if __name__ == "__main__":
    run_ar1()
