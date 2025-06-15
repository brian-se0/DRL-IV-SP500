"""IV environment for SPX ATM-IV one-step forecasting.

This file was migrated from the legacy *src/iv_env.py* implementation and now
lives inside the installable ``econ499`` package so it can be imported with
``from econ499.envs import IVEnv``.

No functional changes were made – only the import path and docstring were
updated.  Downstream agents should depend on this location going forward.
"""

from __future__ import annotations

from typing import List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

__all__ = ["IVEnv", "make_vec"]


class IVEnv(gym.Env):
    """Custom environment for one-step-ahead IV forecasting.

    Parameters
    ----------
    df_slice : pd.DataFrame
        Must contain two columns ``iv_t_orig`` (today's IV) and
        ``iv_t_plus1`` (true next-day IV) plus engineered feature columns.
    feature_list : list[str]
        Exact list/order of columns supplied as observation vector.
    action_scale_factor : float, optional
        Forecast = IVₜ * \(1 + action\_scale\_factor × a\) where ``a`` is the
        agent's action in [-1, 1].  Default 0.1 ⇒ ±10 % moves.
    reward_type : {"mse", "mae"}, optional
        Use negative MSE (default) or negative MAE as reward.
    reward_scale : float, optional
        Multiplicative factor applied to the raw loss (default 1000) so the
        magnitude is comparable to RL defaults.
    arb_penalty_lambda : float, optional
        Arbitrage penalty lambda for static arbitrage penalty.
    penalty_fn : callable | None, optional
        Penalty function for static arbitrage penalty.
    """

    metadata: dict = {}

    def __init__(
        self,
        df_slice: pd.DataFrame,
        feature_list: List[str],
        *,
        maturities: List[int] | None = None,
        action_scale_factor: float = 0.1,
        reward_type: str = "mse",
        reward_scale: float = 1000.0,
        arb_penalty_lambda: float = 0.0,
        penalty_fn: callable | None = None,
    ) -> None:
        super().__init__()
        if reward_type not in {"mse", "mae"}:
            raise ValueError("reward_type must be 'mse' or 'mae'")

        self.df = df_slice.reset_index(drop=True)
        self.feature_list = feature_list
        self.maturities = maturities or [30]
        self.n_maturities = len(self.maturities)
        self.iv_cols = [f"iv_t_orig_{m}" for m in self.maturities]
        self.iv_next_cols = [f"iv_t_plus1_{m}" for m in self.maturities]
        self.action_scale_factor = action_scale_factor
        self.reward_type = reward_type
        self.reward_scale = reward_scale
        self.arb_penalty_lambda = arb_penalty_lambda
        self._penalty_fn = penalty_fn
        self.max_steps = len(self.df) - 1

        self.action_space = spaces.Box(-1, 1, (self.n_maturities,), np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, (len(self.feature_list),), np.float32
        )
        self.current_step = 0

    # ------------------------------------------------------------------
    # core helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        idx = min(self.current_step, len(self.df) - 1)
        return self.df.loc[idx, self.feature_list].to_numpy(np.float32)

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        if self.current_step >= self.max_steps:
            raise IndexError("step() called after episode termination")

        iv_today = self.df.loc[self.current_step, self.iv_cols].to_numpy(np.float32)
        forecast = iv_today * (
            1 + self.action_scale_factor * np.asarray(action, dtype=np.float32)
        )
        actual = self.df.loc[self.current_step, self.iv_next_cols].to_numpy(np.float32)

        if self.reward_type == "mse":
            reward_vec = -self.reward_scale * (forecast - actual) ** 2
        else:  # mae
            reward_vec = -self.reward_scale * np.abs(forecast - actual)
        reward_base = float(np.mean(reward_vec))

        # ---- static arbitrage penalty (optional) ----
        violation = 0.0
        if self.arb_penalty_lambda > 0:
            try:
                if self._penalty_fn is not None:
                    for f in forecast:
                        violation += float(self._penalty_fn(float(f)))
            except Exception:  # pragma: no cover
                violation = 0.0
        reward = reward_base - self.arb_penalty_lambda * max(0.0, violation)

        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        return obs, reward, terminated, truncated, {}


# ----------------------------------------------------------------------
# Convenience vectorised env maker
# ----------------------------------------------------------------------

def make_vec(
    df_slice: pd.DataFrame,
    feature_list: List[str],
    *,
    maturities: List[int] | None = None,
    action_scale_factor: float = 0.1,
    reward_type: str = "mse",
    reward_scale: float = 1000.0,
    arb_penalty_lambda: float = 0.0,
    penalty_fn: callable | None = None,
):
    """Return a `DummyVecEnv` wrapping an :class:`IVEnv`."""

    from stable_baselines3.common.vec_env import DummyVecEnv

    def _init():
        return IVEnv(
            df_slice,
            feature_list,
            maturities=maturities,
            action_scale_factor=action_scale_factor,
            reward_type=reward_type,
            reward_scale=reward_scale,
            arb_penalty_lambda=arb_penalty_lambda,
            penalty_fn=penalty_fn,
        )

    return DummyVecEnv([_init]) 