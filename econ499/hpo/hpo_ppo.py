from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
from stable_baselines3 import PPO

from iv_drl.utils.train_utils import (
    load_and_split_data,
    scale_features,
    create_envs,
    CONFIG,
)
from iv_drl.utils.metrics_utils import rmse

import os
import math

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"]).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The SQLite DB filename will match the study name so parallel studies create
# separate files automatically.
DEFAULT_STUDY_NAME = "ppo_iv_hpo"
STUDY_DB = lambda name: OUTPUT_DIR / f"{name}.db"
BEST_PARAMS_PATH = OUTPUT_DIR / "best_ppo_params.json"

# Short training budget used *within* each Optuna trial.  We only need enough
# steps to differentiate bad configs; the final long training run will use
# many more timesteps.
DEFAULT_TIMESTEPS = 25_000

# -----------------------------------------------------------------------------
#  Optional Stage-2 refinement helpers
# -----------------------------------------------------------------------------


def _load_base_params(path: Optional[str]) -> Optional[dict[str, Any]]:
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


BASE_PARAMS: Optional[dict[str, Any]] = None  # will be set in main()
NARROW_PCT = 0.30  # ±30 % band around centre value


def _narrow_float(trial: optuna.Trial, name: str, low: float, high: float, *, log: bool = False) -> float:
    if BASE_PARAMS and name in BASE_PARAMS:
        center = float(BASE_PARAMS[name])
        span = center * NARROW_PCT
        low = max(low, center - span)
        high = min(high, center + span)
    return trial.suggest_float(name, low, high, log=log)


def _narrow_int(trial: optuna.Trial, name: str, low: int, high: int, *, log: bool = False) -> int:
    orig_low, orig_high = low, high
    if BASE_PARAMS and name in BASE_PARAMS:
        center = int(BASE_PARAMS[name])
        span = max(1, math.ceil(center * NARROW_PCT))
        low = max(low, center - span)
        high = min(high, center + span)
    if low > high:
        low, high = orig_low, orig_high
    return trial.suggest_int(name, low, high, log=log)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _evaluate_model(model: PPO, valid_env) -> float:
    """Roll the model over the validation environment and return RMSE."""

    obs = valid_env.reset()
    done = False
    step = 0
    preds: list[float] = []
    trues: list[float] = []
    # Access underlying IV dataframe to retrieve iv_t columns
    df = valid_env.envs[0].df  # pyright: ignore[reportAttributeAccessIssue]
    scale = valid_env.envs[0].action_scale_factor  # pyright: ignore

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        iv_today = df.loc[step, "iv_t_orig"]
        # ``action`` returned by SB3 can be a NumPy array with shape (n_envs,). Explicitly
        # squeeze to a python float to avoid the "Conversion of an array with ndim > 0 to a
        # scalar is deprecated" warning introduced in NumPy 1.25.
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
# Optuna objective
# -----------------------------------------------------------------------------

def _objective(trial: optuna.Trial) -> float:  # minimise RMSE
    # ------------------------------------------------------------------
    # Fixed search space so Optuna sees the same distribution every
    # trial (avoids "dynamic value space" error).
    #   1. Sample batch_size from a constant categorical list.
    #   2. Sample an integer multiplier k so that n_steps = k * batch_size
    #      remains in [128, 4096].
    # ------------------------------------------------------------------
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    max_k = 4096 // batch_size
    min_k = max(4, 128 // batch_size)
    k = _narrow_int(trial, "rollouts", min_k, max_k, log=True)
    n_steps = batch_size * k

    params: Dict[str, Any] = {
        "n_steps": n_steps,
        "gamma": _narrow_float(trial, "gamma", 0.90, 0.9999),
        "learning_rate": _narrow_float(trial, "learning_rate", 1e-5, 5e-4, log=True),
        "ent_coef": _narrow_float(trial, "ent_coef", 0.0, 0.05),
        "vf_coef": _narrow_float(trial, "vf_coef", 0.1, 1.0),
        "gae_lambda": _narrow_float(trial, "gae_lambda", 0.8, 0.99),
        "clip_range": _narrow_float(trial, "clip_range", 0.1, 0.4),
        "batch_size": batch_size,
    }

    # ------------------------------------------------------------------
    # Data & envs
    # ------------------------------------------------------------------
    train_df, valid_df, feature_cols, categorical_cols, _ = load_and_split_data()
    train_df, valid_df, scaled_feature_cols = scale_features(train_df, valid_df, feature_cols)
    obs_cols = scaled_feature_cols + categorical_cols

    train_env, valid_env = create_envs(
        train_df,
        valid_df,
        obs_cols=obs_cols,
        action_scale_factor=0.05,  # fixed across trials
        reward_type="mse",
        reward_scale=1000,
    )

    # ------------------------------------------------------------------
    # Train short run
    # ------------------------------------------------------------------
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log=None,
        device="cpu",
        **params,
    )
    model.learn(total_timesteps=trial.user_attrs.get("timesteps", DEFAULT_TIMESTEPS))

    # ------------------------------------------------------------------
    # Evaluate on validation env
    # ------------------------------------------------------------------
    val_rmse = _evaluate_model(model, valid_env)

    # Report so Optuna can prune unpromising trials
    trial.report(val_rmse, step=0)
    return val_rmse


# -----------------------------------------------------------------------------
# Public entry-point
# -----------------------------------------------------------------------------

def main(n_trials: int = 50, timesteps: int = DEFAULT_TIMESTEPS, study_name: str = DEFAULT_STUDY_NAME):
    study_db_path = STUDY_DB(study_name)
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=f"sqlite:///{study_db_path}",
        load_if_exists=True,
    )
    # Attach timesteps to trial so objective can access
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def _objective_wrapper(trial):
        trial.user_attrs["timesteps"] = timesteps
        return _objective(trial)

    # Seed the best-known params as the first queued trial so it is always
    # evaluated at the newer (longer) training budget.
    if BASE_PARAMS:
        study.enqueue_trial(BASE_PARAMS)

    study.optimize(_objective_wrapper, n_trials=n_trials, show_progress_bar=True)

    print("Best RMSE:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    BEST_PARAMS_PATH.write_text(json.dumps(study.best_params, indent=2))
    print("Saved best params ->", BEST_PARAMS_PATH)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hyper-parameter search for PPO")
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME, help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Timesteps per trial")
    parser.add_argument("--base-params", type=str, default=None, help="Path to JSON file with seed params")
    args = parser.parse_args()
    BASE_PARAMS = _load_base_params(args.base_params)
    main(n_trials=args.n_trials, timesteps=args.timesteps, study_name=args.study_name)