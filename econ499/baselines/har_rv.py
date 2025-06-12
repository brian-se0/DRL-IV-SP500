from __future__ import annotations

"""HAR-RV baseline for next-day SPX ATM-IV forecasting.

Implements the heterogeneous-autoregressive (HAR) model of Corsi (2009) but
applies it to the daily ATM-IV series, a common proxy for latent volatility in
recent surface-forecasting studies (Gonçalves & Guidolin 2022; Amaya et al.
2022).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

from iv_drl.utils import load_config

CFG = load_config("data_config.yaml")
ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _har_lags(series: pd.Series) -> pd.DataFrame:
    """Return DataFrame with daily, weekly, monthly lag averages.

    The function uses *past* realisations only, ensuring no look-ahead bias.
    """
    df = pd.DataFrame({"d": series.shift(1)})
    df["w"] = series.shift(1).rolling(5).mean()
    df["m"] = series.shift(1).rolling(22).mean()
    return df


# ----------------------------------------------------------------------
# main routine
# ----------------------------------------------------------------------

def run_har_rv(*, out_csv: str | Path | None = None, train_ratio: float = 0.8) -> Path:
    """Fit HAR-RV on ATM-IV series and save out-of-sample forecasts.

    Parameters
    ----------
    out_csv : str | Path | None
        Destination of the forecast CSV.  Defaults to
        ``data_processed/har_rv_oos_predictions.csv``.
    train_ratio : float, optional
        Chronological train / OOS split share (default 0.8).
    """

    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    iv = df[CFG["features"]["target_col"]].astype(float)

    X = _har_lags(iv)
    df_all = pd.concat([df[["date"]], X], axis=1)
    df_all["IV_next"] = iv.shift(-1)
    df_all.dropna(inplace=True)

    split_idx = int(len(df_all) * train_ratio)
    train_df = df_all.iloc[:split_idx]
    oos_df = df_all.iloc[split_idx:]

    y_train = train_df["IV_next"]
    X_train = sm.add_constant(train_df[["d", "w", "m"]])

    model = sm.OLS(y_train, X_train).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Forecast using realised history (no recursion needed)
    X_all = sm.add_constant(df_all[["d", "w", "m"]])
    df_all["har_rv_forecast"] = model.predict(X_all)

    out_path = Path(out_csv) if out_csv else OUT_DIR / "har_rv_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.loc[oos_df.index, ["date", "har_rv_forecast"]].to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )

    # Simple Ljung-Box check on residuals (train set)
    lb_stat, lb_p = acorr_ljungbox(model.resid, lags=[5], return_df=False)
    print("Train Ljung-Box Q(5) p-value:", lb_p[0])
    print('Saved HAR-RV forecasts to', out_path)
    return out_path


if __name__ == "__main__":
    run_har_rv() 