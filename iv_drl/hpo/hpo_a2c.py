from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from stable_baselines3 import A2C

from iv_drl.utils.train_utils import (
    load_and_split_data,
    scale_features,
    create_envs,
)
from iv_drl.utils import load_config
from iv_drl.utils.metrics_utils import rmse

import os
import math
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG = load_config("data_config.yaml")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_STUDY_NAME = "a2c_iv_hpo"
STUDY_DB = lambda name: OUTPUT_DIR / f"{name}.db"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_a2c_params.json"
DEFAULT_TIMESTEPS = 15_000

# -----------------------------------------------------------------------------
# Optional Stage-2 refinement (seed with best params and narrow search space)
# -----------------------------------------------------------------------------

def _load_base_params(path: Optional[str]):
    if not path:
        path = os.getenv("BASE_PARAMS")
    if not path:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists():
        print(f"[WARN] base-params file not found: {p}")
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not load base params from {p}: {exc}")
        return None

BASE_PARAMS: Optional[dict[str, Any]] = None  # set in main()
NARROW_PCT = 0.30

def _narrow_float(trial, name, low, high, *, log=False):
    if BASE_PARAMS and name in BASE_PARAMS:
        center = float(BASE_PARAMS[name])
        span = center * NARROW_PCT
        low = max(low, center - span)
        high = min(high, center + span)
    return trial.suggest_float(name, low, high, log=log)

def _narrow_int(trial, name, low, high, *, log=False):
    orig_low, orig_high = low, high
    if BASE_PARAMS and name in BASE_PARAMS:
        center = int(BASE_PARAMS[name])
        span = max(1, math.ceil(center * NARROW_PCT))
        low = max(low, center - span)
        high = min(high, center + span)
    # If narrowing produced an invalid range, fall back to original bounds
    if low > high:
        low, high = orig_low, orig_high
    return trial.suggest_int(name, low, high, log=log)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _evaluate_model(model: A2C, valid_env) -> float:
    obs = valid_env.reset()
    done = False
    step = 0
    preds: list[float] = []
    trues: list[float] = []
    df = valid_env.envs[0].df  # type: ignore[attr-defined]
    scale = valid_env.envs[0].action_scale_factor  # type: ignore[attr-defined]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        iv_today = df.loc[step, "iv_t_orig"]
        action_scalar = float(np.asarray(action).squeeze())
        forecast = iv_today * (1 + scale * action_scalar)
        actual = df.loc[step, "iv_t_plus1"]
        preds.append(forecast)
        trues.append(actual)
        obs, _rewards, done_vec, _info = valid_env.step(action)
        done = bool(done_vec[0])
        step += 1

    return rmse(np.asarray(trues), np.asarray(preds))


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------

def _objective(trial: optuna.Trial) -> float:
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    max_k = 1024 // batch_size
    min_k = max(2, 64 // batch_size)
    k = _narrow_int(trial, "rollouts", min_k, max_k, log=True)
    n_steps = batch_size * k

    params: Dict[str, Any] = {
        "n_steps": n_steps,
        "gamma": _narrow_float(trial, "gamma", 0.90, 0.9999),
        "learning_rate": _narrow_float(trial, "learning_rate", 1e-5, 1e-3, log=True),
        "ent_coef": _narrow_float(trial, "ent_coef", 0.0, 0.05),
        "vf_coef": _narrow_float(trial, "vf_coef", 0.1, 1.0),
        # batch_size is not a valid A2C kwarg; only used to compute n_steps.
    }

    train_df, valid_df, feature_cols, categorical_cols, _ = load_and_split_data()
    train_df, valid_df, scaled_feature_cols = scale_features(train_df, valid_df, feature_cols)
    obs_cols = scaled_feature_cols + categorical_cols

    train_env, valid_env = create_envs(
        train_df,
        valid_df,
        obs_cols=obs_cols,
        action_scale_factor=0.05,
        reward_type="mse",
        reward_scale=1000,
    )

    model = A2C(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=None,
        device="cpu",
        **params,
    )
    model.learn(total_timesteps=trial.user_attrs.get("timesteps", DEFAULT_TIMESTEPS))

    return _evaluate_model(model, valid_env)


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------

def main(n_trials: int = 50, timesteps: int = DEFAULT_TIMESTEPS, study_name: str = DEFAULT_STUDY_NAME):
    study_db_path = STUDY_DB(study_name)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=True,
    )

    def _wrapper(trial):
        trial.user_attrs["timesteps"] = timesteps
        return _objective(trial)

    if BASE_PARAMS:
        study.enqueue_trial(BASE_PARAMS)

    study.optimize(_wrapper, n_trials=n_trials, show_progress_bar=True)

    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print("Best RMSE:", study.best_value)
    print("Saved best params ->", BEST_PARAMS_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyper-parameter search for A2C")
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--base-params", type=str, default=None)
    args = parser.parse_args()
    BASE_PARAMS = _load_base_params(args.base_params)
    main(n_trials=args.n_trials, timesteps=args.timesteps, study_name=args.study_name) 