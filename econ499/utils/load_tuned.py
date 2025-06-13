from __future__ import annotations

"""Utility to load the *tuned* hyper-parameter JSON files and (optionally) fix RNG seeds.

This helper centralises two common steps performed by training scripts:
1.  Reading the Optuna best-trial dictionary from ``config/tuned/{algo}.json``.
2.  Seeding Python, NumPy, PyTorch and Stable-Baselines3 so results are
    *exactly* reproducible when ``seed`` is provided.

Example
-------
>>> from econ499.utils.load_tuned import load_tuned_params
>>> ppo_params = load_tuned_params('ppo', seed=42)
>>> model = PPO('MlpPolicy', env, **ppo_params)
"""

from pathlib import Path
import json
import random
from typing import Any, Dict, Final, Literal, Optional

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
#   project_root / config / tuned / ppo.json
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
TUNED_DIR: Final[Path] = PROJECT_ROOT / "config" / "tuned"

# Public algorithms we ship tuned parameters for
AlgoName = Literal["ppo", "a2c", "lstm"]

# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch and SB3.

    This is a thin wrapper around the respective libraries' seeding utilities so
    that *all* stochasticity is locked down.  For GPU determinism we also
    disable CuDNN benchmarking.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Ensure deterministic CuDNN kernel selection (has a small perf hit)
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    # SB3 uses Gymnasium / numpy RNGs internally; calling its helper will in
    # turn set the global Gym RNG as well.
    set_random_seed(seed)


def load_tuned_params(algo: AlgoName, *, seed: Optional[int] = None, param_dir: str | Path | None = None) -> Dict[str, Any]:
    """Return the tuned hyper-parameters for *algo*.

    Parameters
    ----------
    algo : {"ppo", "a2c", "lstm"}
        The algorithm whose JSON file should be loaded.
    seed : int, optional
        If given, calls ``_seed_everything`` and *adds* ``{"seed": seed}`` to
        the returned dict (ignored for LSTM, which handles seeds internally).
    param_dir : str | Path, optional
        Override the directory that contains the ``{algo}.json`` files.  By
        default this is ``PROJECT_ROOT/config/tuned``.

    Returns
    -------
    dict
        A copy of the JSON dict ready to unpack as keyword-arguments when
        instantiating the model.
    """

    algo = algo.lower()  # normalise capitalisation
    if algo not in ("ppo", "a2c", "lstm"):
        raise ValueError(f"Unsupported algo '{algo}'. Expected one of 'ppo', 'a2c', 'lstm'.")

    base_dir = Path(param_dir) if param_dir else TUNED_DIR
    json_path = base_dir / f"{algo}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Tuned-parameter file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        params: Dict[str, Any] = json.load(fh)

    # SB3 agents pass seed via the constructor.  The LSTM baseline seeds inside
    # its training loop so we skip injecting the key for that specific model.
    if seed is not None and algo in {"ppo", "a2c"}:
        _seed_everything(seed)
        params = {**params, "seed": seed}

    # The Optuna search stores "rollouts" so that n_steps = rollouts * batch_size
    # Convert it once here so training scripts receive a *direct* SB3 kwarg.
    if algo in {"ppo", "a2c"} and "rollouts" in params and "batch_size" in params:
        params = params.copy()  # avoid mutating the JSON representation
        params["n_steps"] = params["batch_size"] * params.pop("rollouts")

    return params

__all__ = [
    "load_tuned_params",
] 