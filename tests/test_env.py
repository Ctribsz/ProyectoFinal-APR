"""Pruebas de BalanceBotEnv (entregable E2, tarea T6 de docs/fase-1-entorno.md)."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import BalanceBotEnv, load_config
from envs.balance_bot_env import REWARD_KEYS


@pytest.fixture
def env() -> BalanceBotEnv:
    e = BalanceBotEnv(level=0)
    yield e
    e.close()


def _set_pitch(env: BalanceBotEnv, theta: float) -> None:
    import mujoco

    q = env.model.joint("root").qposadr[0]
    env.data.qpos[q + 3 : q + 7] = [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0]
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def test_check_env(env: BalanceBotEnv) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env, skip_render_check=True)


def test_gym_make() -> None:
    import gymnasium as gym

    e = gym.make("BalanceBot-v0", level=0)
    obs, _ = e.reset(seed=0)
    assert e.observation_space.contains(obs)
    e.close()


def test_determinism() -> None:
    rng = np.random.default_rng(123)
    actions = rng.uniform(-1, 1, size=(200, 2)).astype(np.float32)
    trajectories = []
    for _ in range(2):
        e = BalanceBotEnv(level=0)
        obs, _ = e.reset(seed=7)
        traj = [obs]
        for a in actions:
            obs, _, terminated, truncated, _ = e.step(a)
            traj.append(obs)
            if terminated or truncated:
                break
        trajectories.append(np.array(traj))
    np.testing.assert_array_equal(trajectories[0], trajectories[1])


def test_reset_seeds_differ(env: BalanceBotEnv) -> None:
    pitches = {round(float(env.reset(seed=s)[0][0]), 6) for s in range(10)}
    assert len(pitches) == 10
    assert all(abs(p) <= env.initial_pitch_range for p in pitches)


def test_obs_valid(env: BalanceBotEnv) -> None:
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        assert obs.shape == (9,) and obs.dtype == np.float32
        assert np.isfinite(obs).all()
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()


def test_obs_layout(env: BalanceBotEnv) -> None:
    """Comandos en cero en el nivel 0 y la acción anterior se refleja en la observación."""
    env.reset(seed=0)
    action = np.array([0.25, -0.5], dtype=np.float32)
    obs, *_ = env.step(action)
    np.testing.assert_array_equal(obs[5:7], [0.0, 0.0])
    np.testing.assert_allclose(obs[7:9], action)


def test_action_is_clipped(env: BalanceBotEnv) -> None:
    env.reset(seed=0)
    obs, *_ = env.step(np.array([5.0, -5.0], dtype=np.float32))
    np.testing.assert_allclose(env.data.ctrl, [1.0, -1.0])
    np.testing.assert_allclose(obs[7:9], [1.0, -1.0])


def test_falls_without_control(env: BalanceBotEnv) -> None:
    env.reset(seed=0)
    for step in range(env.max_episode_steps):
        _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
        if terminated:
            break
    assert terminated and not truncated
    assert step < env.max_episode_steps - 1


def test_termination_threshold(env: BalanceBotEnv) -> None:
    env.reset(seed=0)
    _set_pitch(env, 0.55)
    crossed = False
    for _ in range(100):
        obs, _, terminated, _, info = env.step(np.zeros(2, dtype=np.float32))
        assert terminated == (abs(info["pitch"]) > env.termination_pitch)
        assert obs[0] == pytest.approx(info["pitch"], abs=1e-6)
        if terminated:
            crossed = True
            break
    assert crossed


def test_truncation() -> None:
    cfg = load_config(0)
    cfg["max_episode_steps"] = 5
    cfg["initial_pitch_range"] = 0.0
    e = BalanceBotEnv(level=0, config=cfg)
    e.reset(seed=0)
    for i in range(5):
        _, _, terminated, truncated, _ = e.step(np.zeros(2, dtype=np.float32))
        assert not terminated
        assert truncated == (i == 4)


def test_reward_components(env: BalanceBotEnv) -> None:
    env.reset(seed=0)
    for _ in range(50):
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        comps = info["reward_components"]
        assert set(comps) == set(REWARD_KEYS)
        expected = sum(env.reward_weights[k] * comps[k] for k in REWARD_KEYS)
        assert reward == pytest.approx(expected)
        if terminated or truncated:
            break


def test_pitch_sign(env: BalanceBotEnv) -> None:
    """Acción positiva en ambas ruedas: el robot avanza y el chasis se inclina hacia atrás (θ < 0, §4.1)."""
    env.reset(seed=0)
    _set_pitch(env, 0.0)
    for _ in range(5):
        obs, _, _, _, info = env.step(np.array([0.3, 0.3], dtype=np.float32))
    assert obs[0] < 0
    assert obs[3] > 0 and obs[4] > 0
    assert info["v"] > 0


def test_forward_lean_increases_pitch(env: BalanceBotEnv) -> None:
    env.reset(seed=0)
    _set_pitch(env, 0.02)
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert obs[0] > 0.02 and obs[1] > 0


def test_invalid_level_and_mismatched_config() -> None:
    with pytest.raises(ValueError):
        BalanceBotEnv(level=4)
    with pytest.raises(ValueError):
        BalanceBotEnv(level=1, config=load_config(0))
