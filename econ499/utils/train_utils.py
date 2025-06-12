"""Shared helpers for training DRL agents.

Migrated from the legacy *src/train_utils.py* so that other modules can simply
``from iv_drl.utils.train_utils import load_and_split_data``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import load_config
from iv_drl.envs import make_vec

CONFIG = load_config("data_config.yaml")
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
DATA_CSV = Path(CONFIG["paths"]["drl_state_file"])
TARGET_IV_COL = CONFIG["features"]["target_col"]


def load_and_split_data(*, exclude_blocks: List[str] | None = None) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str], str]:
    """Load processed panel and split into train/validation chronologically.

    Parameters
    ----------
    exclude_blocks : list[str] | None
        Names of feature blocks to drop (defined in data_config.yaml → feature_blocks).
    """

    feature_cols = CONFIG["features"]["all_feature_cols"].copy()

    # Drop blocks if requested
    if exclude_blocks:
        blocks = CONFIG.get("feature_blocks", {})
        drop_cols = {c for b in exclude_blocks for c in blocks.get(b, [])}
        feature_cols = [c for c in feature_cols if c not in drop_cols]

    df = (
        pd.read_csv(DATA_CSV, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    needed_cols = [TARGET_IV_COL] + feature_cols + CONFIG["features"]["categorical_cols"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        print(
            f"[WARN] Dropping {len(missing)} missing feature columns not present in {DATA_CSV}: {missing}"
        )
        feature_cols = [c for c in feature_cols if c not in missing]
        needed_cols = [TARGET_IV_COL] + feature_cols + CONFIG["features"]["categorical_cols"]

    df["iv_t_orig"] = df[TARGET_IV_COL]
    df["iv_t_plus1"] = df[TARGET_IV_COL].shift(-1)
    df.dropna(subset=["iv_t_orig", "iv_t_plus1"] + feature_cols, inplace=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()

    return train_df, valid_df, feature_cols, CONFIG["features"]["categorical_cols"], TARGET_IV_COL


def scale_features(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Standard-scale *feature_cols* and return new column names."""

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    scaled_cols = [f"{c}_scaled" for c in feature_cols]
    train_df.loc[:, scaled_cols] = scaler.transform(train_df[feature_cols])
    valid_df.loc[:, scaled_cols] = scaler.transform(valid_df[feature_cols])
    return train_df, valid_df, scaled_cols


def create_envs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    obs_cols: List[str],
    *,
    action_scale_factor: float = 0.1,
    reward_type: str = "mse",
    reward_scale: float = 1000.0,
    arb_penalty_lambda: float = 0.0,
    penalty_fn: callable | None = None,
):
    """Return training and validation vectorised Gym environments."""

    train_env = make_vec(
        train_df,
        obs_cols,
        action_scale_factor=action_scale_factor,
        reward_type=reward_type,
        reward_scale=reward_scale,
        arb_penalty_lambda=arb_penalty_lambda,
        penalty_fn=penalty_fn,
    )
    valid_env = make_vec(
        valid_df,
        obs_cols,
        action_scale_factor=action_scale_factor,
        reward_type=reward_type,
        reward_scale=reward_scale,
        arb_penalty_lambda=arb_penalty_lambda,
        penalty_fn=penalty_fn,
    )
    return train_env, valid_env


__all__ = [
    "load_and_split_data",
    "scale_features",
    "create_envs",
    "CONFIG",
] 