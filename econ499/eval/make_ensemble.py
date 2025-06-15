from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from econ499.utils import load_config
from econ499.eval.evaluate_all import _find_forecast_col

CFG = load_config("data_config.yaml")
DATA_DIR = Path(CFG["paths"]["output_dir"]).resolve()


def make_ensemble(csv_paths: Sequence[str | Path], *, out_csv: str | Path | None = None) -> Path:
    """Average predictions from multiple CSV files.

    Parameters
    ----------
    csv_paths : Sequence[str | Path]
        Paths of *_oos_predictions.csv* files to average.
    out_csv : str | Path | None, optional
        Destination CSV (defaults to ``DATA_DIR/ensemble_oos_predictions.csv``).
    """
    dfs: list[pd.DataFrame] = []
    for p in csv_paths:
        path = Path(p).expanduser().resolve()
        df = pd.read_csv(path, parse_dates=["date"])
        col = _find_forecast_col(df)
        dfs.append(df[["date", col]].rename(columns={col: path.stem}))

    if not dfs:
        raise ValueError("No prediction CSVs provided")

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="date", how="inner")

    merged["ensemble_forecast"] = merged.drop(columns=["date"]).mean(axis=1)

    out_path = Path(out_csv) if out_csv else DATA_DIR / "ensemble_oos_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged[["date", "ensemble_forecast"]].to_csv(out_path, index=False, float_format="%.6f")

    print("Saved ensemble forecasts to", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Average forecast CSVs")
    parser.add_argument("csvs", nargs="+", help="Input *_oos_predictions.csv files")
    parser.add_argument("--out", type=str, default=None, help="Destination CSV path")
    args = parser.parse_args()
    make_ensemble(args.csvs, out_csv=args.out)


if __name__ == "__main__":
    main()
