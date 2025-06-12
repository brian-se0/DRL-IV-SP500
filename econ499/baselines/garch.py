"""GARCH(1,1) baseline migrated from legacy script."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from arch import arch_model
import warnings

from iv_drl.utils import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*found in sys.modules.*")

CFG = load_config("data_config.yaml")
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = (REPO_ROOT / CFG["paths"]["drl_state_file"]).resolve()
OUT_DIR = Path(CFG["paths"]["output_dir"]).resolve()

def run_garch(*, out_csv: str | Path | None = None, train_ratio: float = 0.8) -> Path:
    """Fit AR(1)-GARCH(1,1) on ATM IV series and save OOS forecasts."""

    df = pd.read_csv(DATA_CSV, parse_dates=["date"]).sort_values("date")
    SCALE = 100.0  # rescale factor to bring series closer to order(1)
    iv = df[CFG["features"]["target_col"]].astype(float) * SCALE

    split_idx = int(len(iv) * train_ratio)
    y_train = iv.iloc[:split_idx]
    y_oos = iv.iloc[split_idx:]

    am = arch_model(y_train, mean="AR", lags=1, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")

    forecasts = res.forecast(start=y_train.index[-1], horizon=len(y_oos), reindex=False)
    preds = forecasts.mean.iloc[1:, 0].to_numpy()[: len(y_oos)] / SCALE

    out_path = Path(out_csv) if out_csv else OUT_DIR / "garch_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": df["date"].iloc[split_idx:][: len(preds)], "garch_forecast": preds}).to_csv(
        out_path, index=False, date_format="%Y-%m-%d"
    )
    print('Saved GARCH forecasts to', out_path)
    return out_path

if __name__ == "__main__":
    run_garch() 