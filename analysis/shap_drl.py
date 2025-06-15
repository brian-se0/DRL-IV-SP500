from __future__ import annotations

"""Compute SHAP values for a trained DRL model.

This utility loads a saved PPO/A2C model, samples observations from the
validation slice, and computes feature attributions using SHAP.  A bar
plot of mean absolute SHAP values is saved under ``artifacts/figures``.

Example
-------
>>> python analysis/shap_drl.py --model results/ppo_best_model/best_model.zip
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C

from econ499.utils.train_utils import load_and_split_data, scale_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(path: Path):
    """Load a saved SB3 model and infer algorithm."""
    if "ppo" in path.stem.lower():
        return PPO.load(path, device="cpu")
    if "a2c" in path.stem.lower():
        return A2C.load(path, device="cpu")
    for cls in (PPO, A2C):
        try:
            return cls.load(path, device="cpu")
        except Exception:
            continue
    raise ValueError(f"Unsupported model file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SHAP values for a DRL model")
    parser.add_argument("--model", type=str, required=True, help="Path to saved SB3 model")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of validation observations")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        parser.error(f"Model file not found: {model_path}")

    model = _load_model(model_path)

    # ------------------------------------------------------------------
    # Load data exactly as during training
    # ------------------------------------------------------------------
    train_df, valid_df, feature_cols, cat_cols, _ = load_and_split_data()
    train_df, valid_df, scaled_cols = scale_features(train_df, valid_df, feature_cols)
    obs_cols = scaled_cols + cat_cols
    valid_df = valid_df.reset_index(drop=True)

    background = valid_df[obs_cols].iloc[:50]
    sample = valid_df[obs_cols].iloc[: args.n_samples]

    def predict(obs: np.ndarray) -> np.ndarray:
        actions, _ = model.predict(obs, deterministic=True)
        return np.asarray(actions).reshape(-1)

    explainer = shap.Explainer(predict, background)
    shap_values = explainer(sample)

    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(obs_cols)), mean_abs[order])
    plt.xticks(range(len(obs_cols)), np.array(obs_cols)[order], rotation=45, ha="right")
    plt.ylabel("Mean |SHAP value|")
    plt.tight_layout()

    out_path = FIG_DIR / f"{model_path.stem}_shap_bar.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[FIG] SHAP bar plot -> {out_path}")


if __name__ == "__main__":
    main()
