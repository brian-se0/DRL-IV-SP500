import numpy as np
import pandas as pd
import pytest

from econ499.envs.iv_env import IVEnv


def test_ivenv_step_rewards_and_done():
    df = pd.DataFrame(
        {
            "iv_t_orig_30": [1.0, 2.0, 3.0],
            "iv_t_plus1_30": [1.1, 1.9, 3.0],
            "feat": [0.0, 0.0, 0.0],
        }
    )
    env = IVEnv(df, ["feat"], maturities=[30], reward_scale=1.0)
    obs, info = env.reset()

    obs1, reward1, term1, trunc1, _ = env.step(np.array([0.0]))
    assert reward1 == pytest.approx(-0.01)
    assert not term1
    assert not trunc1
    assert env.current_step == 1
    assert obs1 == pytest.approx(np.array([0.0], dtype=np.float32))

    obs2, reward2, term2, trunc2, _ = env.step(np.array([0.0]))
    assert reward2 == pytest.approx(-0.01)
    assert term2
    assert not trunc2
    assert env.current_step == 2
    assert np.all(obs2 == 0)

    with pytest.raises(IndexError):
        env.step(np.array([0.0]))
