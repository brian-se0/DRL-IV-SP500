"""Generate out-of-sample forecasts with a trained DRL model.

Usage (from repo root):

    python -m iv_drl.predict.make_drl_forecast \
           --model data_processed/ppo_best_model.zip \
           --out   data_processed/ppo_oos_predictions.csv

The script loads the saved SB3 model, re-creates the validation environment
exactly as during training, and rolls the agent forward deterministically to
produce a 1-day-ahead IV forecast for each observation in the validation slice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
# Lazy import of algo classes for loading
from stable_baselines3 import PPO, A2C

from iv_drl.utils.train_utils import (
    load_and_split_data,
    scale_features,
    create_envs,
    CONFIG,
)

OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"]).resolve()

def _infer_col_name(model_path: Path, suffix: str | None = None) -> str:
    """Return forecast column name like ppo_macro_forecast."""
    # Detect algorithm name from any part of the path (directory or file name)
    base = "drl"
    for part in model_path.parts:
        lower = part.lower()
        if "ppo" in lower:
            base = "ppo"
            break
        if "a2c" in lower:
            base = "a2c"
            break

    return f"{base}_{suffix}_forecast" if suffix else f"{base}_forecast"

def _load_sb3_model(model_path: Path, device: str | None = None):
    """Load a saved SB3 model using the appropriate algorithm class."""
    lower = model_path.stem.lower()
    if "ppo" in lower:
        return PPO.load(model_path, device=device or 'auto', print_system_info=False)
    if "a2c" in lower:
        return A2C.load(model_path, device=device or 'auto', print_system_info=False)

    # Fallback: try each supported class
    for cls in (PPO, A2C):
        try:
            return cls.load(model_path, device=device or 'auto', print_system_info=False)
        except Exception:
            continue
    raise ValueError(f"Unsupported or corrupted model file: {model_path}")

def make_forecast(model_path: Path, out_csv: Path, *, exclude_blocks: list[str] | None = None, device: str | None = None, suffix: str | None = None) -> Path:
    # ------------------------------------------------------------------
    # Load data and model
    # ------------------------------------------------------------------
    model = _load_sb3_model(model_path, device=device)

    train_df, valid_df, feature_cols, categorical_cols, _ = load_and_split_data(exclude_blocks=exclude_blocks)
    train_df, valid_df, scaled_feature_cols = scale_features(train_df, valid_df, feature_cols)
    obs_cols = scaled_feature_cols + categorical_cols

    # Re-create envs exactly as during training
    _train_env, valid_env = create_envs(
        train_df,
        valid_df,
        obs_cols=obs_cols,
        action_scale_factor=0.05,  # must match training
        reward_type="mse",
        reward_scale=1000,
    )

    # ------------------------------------------------------------------
    # Roll forward deterministically
    # ------------------------------------------------------------------
    obs = valid_env.reset()
    done = False
    step = 0
    preds: list[float] = []
    dates: list[pd.Timestamp] = []
    df = valid_env.envs[0].df  # type: ignore[attr-defined]
    scale = valid_env.envs[0].action_scale_factor  # type: ignore[attr-defined]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        iv_today = df.loc[step, "iv_t_orig"]
        action_scalar = float(np.asarray(action).squeeze())
        forecast = iv_today * (1 + scale * action_scalar)

        preds.append(forecast)
        dates.append(df.loc[step, "date"])

        obs, _rewards, done_vec, _info = valid_env.step(action)
        done = bool(done_vec[0])
        step += 1

    # determine column suffix priority: explicit > exclude_block > None
    _sfx = suffix if suffix is not None else (exclude_blocks[0] if exclude_blocks else None)
    col = _infer_col_name(model_path, suffix=_sfx)
    out_df = pd.DataFrame({"date": dates, col: preds})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, float_format="%.6f")

    print(f"Saved {len(out_df):,} forecasts -> {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Generate OOS forecasts with a DRL model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved SB3 .zip model")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Destination CSV (defaults to data_processed/<stem>_oos_predictions.csv)",
    )
    parser.add_argument('--exclude_block', nargs='*', help='Feature block(s) to drop, align with training')
    parser.add_argument('--device', type=str, default='cpu', help='Device for inference cpu|cuda|auto')
    parser.add_argument('--suffix', type=str, default=None, help='Custom suffix for forecast column')
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()

    if model_path.is_dir():
        candidate = model_path / "best_model.zip"
        if candidate.exists():
            model_path = candidate
        else:
            # fallback to first zip in dir
            zips = list(model_path.glob('*.zip'))
            if zips:
                model_path = zips[0]
            else:
                parser.error(f"No .zip model found inside {model_path}")

    if not model_path.exists():
        parser.error(f"model file not found: {model_path}")

    if args.out:
        out_csv = Path(args.out).resolve()
    else:
        default_name = f"{model_path.stem}_oos_predictions.csv"
        out_csv = OUTPUT_DIR / default_name

    make_forecast(model_path, out_csv, exclude_blocks=args.exclude_block, device=args.device, suffix=args.suffix)


if __name__ == "__main__":
    main() 