"""PPO training entry point (migrated from src/ppo_iv_forecast.py)."""
from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecMonitor
import yaml

from econ499.utils.train_utils import (
    load_and_split_data,
    scale_features,
    create_envs,
    CONFIG,
)
from econ499.utils import load_tuned_params

OUTPUT_DIR = Path(CONFIG["paths"]["output_dir"])
TENSORBOARD_LOG_DIR = OUTPUT_DIR / "tensorboard_logs_ppo"
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)


def train_ppo(total_timesteps: int = 100_000, action_scale: float = 0.05, *, exclude_blocks: list[str] | None = None, arb_lambda: float = 0.0, hparam_file: str | None = None, seed: int | None = None) -> Path:
    """Train PPO agent and return path to the best model zip."""

    # Load and prep data
    train_df, valid_df, feature_cols, categorical_cols, _ = load_and_split_data(exclude_blocks=exclude_blocks)
    train_df, valid_df, scaled_feature_cols = scale_features(train_df, valid_df, feature_cols)
    obs_cols = scaled_feature_cols + categorical_cols

    train_env, valid_env = create_envs(
        train_df,
        valid_df,
        obs_cols=obs_cols,
        maturities=CONFIG["features"]["surface"]["maturities"],
        action_scale_factor=action_scale,
        reward_type="mse",
        reward_scale=1000,
        arb_penalty_lambda=arb_lambda,
    )

    from stable_baselines3.common.vec_env import VecMonitor as _VM
    train_env = _VM(train_env)

    eval_env = VecMonitor(valid_env)
    suffix = "" if arb_lambda == 0 else f"_l{int(arb_lambda)}"
    best_path = OUTPUT_DIR / f"ppo_best_model{suffix}"
    # Early-stopping when validation RMSE has not improved for `patience` evals
    stopper = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,  # patience
        min_evals=20,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_path),
        log_path=str(OUTPUT_DIR / "ppo_eval_logs"),
        eval_freq=1000,
        deterministic=True,
        callback_after_eval=stopper,
    )

    try:
        tb_log = str(TENSORBOARD_LOG_DIR)
        import tensorboard  # noqa: F401
    except ImportError:
        tb_log = None

    # ------------------------------------------------------------------
    # Load tuned hyper-parameters and merge with safe defaults.
    # ------------------------------------------------------------------
    try:
        tuned = load_tuned_params("ppo", seed=seed)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Could not load tuned params via helper: {exc}")
        tuned = {}

    # Merge hard-coded defaults with tuned params and YAML overrides
    default_kwargs = dict(
        gamma=0.99,
        n_steps=2048,
        ent_coef=0.01,
        learning_rate=3e-4,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        batch_size=64,
    )

    # Pull Optuna-tuned values in (overwriting defaults)
    default_kwargs.update(tuned)

    # Optionally override defaults via YAML file (highest priority)
    if hparam_file:
        try:
            with open(hparam_file, "r", encoding="utf-8") as fh:
                yaml_params: dict = yaml.safe_load(fh) or {}
                default_kwargs.update(yaml_params)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Could not read hparam_file {hparam_file}: {exc}")

    # Always train on CPU; GPU gives no speed-up for MLP policies.
    kwargs_extra = {"device": "cpu"}

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=tb_log,
        **default_kwargs,
        **kwargs_extra,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    return best_path.with_suffix(".zip")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO for SPX ATM-IV forecasting")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Max training timesteps (upper bound)")
    parser.add_argument("--exclude_block", nargs="*", help="Feature block(s) to drop: surface realised macro")
    parser.add_argument("--arb_lambda", type=float, default=0.0, help="Static-arbitrage penalty lambda")
    parser.add_argument("--hparam_file", type=str, default=None, help="YAML file with SB3 hyper-parameters")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed (set 42 for full reproducibility)")
    args = parser.parse_args()

    print(f"Training PPO for up to {args.timesteps} steps with early-stopping")
    path = train_ppo(total_timesteps=args.timesteps, exclude_blocks=args.exclude_block, arb_lambda=args.arb_lambda, hparam_file=args.hparam_file, seed=args.seed)
    print("Best model saved to:", path) 