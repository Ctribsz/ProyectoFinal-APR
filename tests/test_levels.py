"""Pruebas de los niveles 1–3: perturbaciones, comandos y aleatorización (entregable E8)."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from baselines.pid_controller import PIDController, run_episodes
from envs import BalanceBotEnv, load_config
from envs.randomization import SensorCorruptor

ZERO = np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_check_env_levels(level: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=".*Not able to test alternative render modes.*")
        check_env(BalanceBotEnv(level=level), skip_render_check=True)


def test_same_observation_space_all_levels() -> None:
    spaces = [BalanceBotEnv(level=lv).observation_space for lv in range(4)]
    assert all(s == spaces[0] for s in spaces)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_determinism_levels(level: int) -> None:
    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.3, 0.3, size=(300, 2)).astype(np.float32)
    trajs = []
    for _ in range(2):
        env = BalanceBotEnv(level=level)
        obs, _ = env.reset(seed=11)
        traj = [obs]
        for a in actions:
            obs, *_ , terminated, truncated, _ = env.step(a)
            traj.append(obs)
            if terminated or truncated:
                break
        trajs.append(np.array(traj))
    np.testing.assert_array_equal(trajs[0], trajs[1])


@pytest.mark.parametrize("level", [1, 2, 3])
def test_obs_within_bounds(level: int) -> None:
    env = BalanceBotEnv(level=level)
    obs, _ = env.reset(seed=0)
    for _ in range(500):
        assert env.observation_space.contains(obs)
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, _ = env.reset()


# ---------------------------------------------------------------- perturbaciones


def test_scheduled_push_timing_and_direction() -> None:
    env = BalanceBotEnv(level=0)  # sin empujones aleatorios
    env.reset(seed=0)
    env.perturbations.schedule(start=0.5, force=8.0, angle=0.0)
    chassis = env.model.body("chassis").id
    com_x0 = env.data.subtree_com[chassis, 0]
    active_steps, com_x_after = 0, None
    for _ in range(100):
        _, _, _, _, info = env.step(ZERO)
        if info["push_active"]:
            active_steps += 1
            np.testing.assert_allclose(np.linalg.norm(env.data.xfrc_applied[chassis, :3]), 8.0)
        elif active_steps and com_x_after is None:
            com_x_after = env.data.subtree_com[chassis, 0]
    assert active_steps == round(env.perturbations.duration / env.dt)
    assert not env.data.xfrc_applied.any()
    # El centro de masa avanza (+x). El eje de las ruedas puede retroceder: el empujón alto inclina el cuerpo.
    assert com_x_after > com_x0


def test_push_tilts_backward_when_pushed_forward() -> None:
    """El empujón en la parte alta del chasis genera torque de cabeceo: empujado hacia adelante, θ crece."""
    env = BalanceBotEnv(level=0)
    env.reset(seed=0)
    env.perturbations.schedule(start=0.0, force=8.0, angle=0.0)
    obs, *_ = env.step(ZERO)
    assert obs[1] > 0


def test_random_pushes_rate() -> None:
    env = BalanceBotEnv(level=1)
    env.reset(seed=0)
    starts, was_active = 0, False
    pid = PIDController.from_env(env)
    obs, _ = env.reset(seed=0)
    for _ in range(3):
        for _ in range(env.max_episode_steps):
            action, _ = pid.predict(obs)
            obs, _, terminated, truncated, info = env.step(action)
            starts += info["push_active"] and not was_active
            was_active = info["push_active"]
            if terminated or truncated:
                obs, _ = env.reset()
                pid.reset()
                was_active = False
                break
    # 30 s con un intervalo medio de ~2.1 s → ~14 empujones; rango amplio para no ser frágil.
    assert 6 <= starts <= 24


def test_no_pushes_level0() -> None:
    env = BalanceBotEnv(level=0)
    env.reset(seed=0)
    for _ in range(500):
        *_, info = env.step(ZERO)
        assert not info["push_active"]
        if abs(info["pitch"]) > 0.6:
            break


# ---------------------------------------------------------------- comandos


def test_commands_level2() -> None:
    env = BalanceBotEnv(level=2)
    cfg = env.config["commands"]
    obs, _ = env.reset(seed=3)
    pid = PIDController.from_env(env)
    changes, last, last_change_t = [], obs[5:7].copy(), 0.0
    for _ in range(env.max_episode_steps):
        action, _ = pid.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        np.testing.assert_allclose(obs[5:7], info["command"], atol=1e-6)
        assert abs(obs[5]) <= cfg["v_max"] and abs(obs[6]) <= cfg["yaw_rate_max"]
        if not np.allclose(obs[5:7], last):
            changes.append(env.data.time - last_change_t)
            last_change_t, last = env.data.time, obs[5:7].copy()
        if terminated or truncated:
            break
    assert changes, "el comando nunca cambió en 10 s"
    assert all(cfg["min_interval"] - env.dt <= c <= cfg["max_interval"] + env.dt for c in changes)


@pytest.mark.parametrize("level", [0, 1])
def test_commands_zero_without_commands(level: int) -> None:
    env = BalanceBotEnv(level=level)
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        assert obs[5] == 0 and obs[6] == 0
        obs, _, terminated, truncated, _ = env.step(ZERO)
        if terminated or truncated:
            break


# ---------------------------------------------------------------- aleatorización


def test_domain_randomization_ranges_and_variation() -> None:
    env = BalanceBotEnv(level=3)
    cfg = env.config["randomization"]
    chassis = env.model.body("chassis").id
    nominal_mass = BalanceBotEnv(level=0).model.body_mass[chassis]
    masses = []
    for seed in range(10):
        _, info = env.reset(seed=seed)
        f = info["randomization"]
        assert abs(f["mass"] - 1) <= cfg["mass_range"]
        assert abs(f["friction"] - 1) <= cfg["friction_range"]
        assert abs(f["torque"] - 1) <= cfg["torque_range"]
        np.testing.assert_allclose(env.model.body_mass[chassis], nominal_mass * f["mass"])
        np.testing.assert_allclose(env.model.actuator_gear[:, 0], 0.3 * f["torque"])
        masses.append(env.model.body_mass[chassis])
        assert 0 <= info["latency"] <= cfg["max_latency"]
    assert len(set(np.round(masses, 6))) == 10


def test_no_randomization_below_level3() -> None:
    for level in (0, 1, 2):
        _, info = BalanceBotEnv(level=level).reset(seed=0)
        assert info["randomization"] == {"mass": 1.0, "friction": 1.0, "torque": 1.0}
        assert info["latency"] == 0


def test_sensor_latency() -> None:
    corruptor = SensorCorruptor({"enabled": True, "max_latency": 2, "noise": {}})
    rng = np.random.default_rng(0)
    corruptor.reset(rng, np.zeros(5))
    corruptor.latency = 2
    corruptor._buffer = type(corruptor._buffer)([np.zeros(5)] * 3, maxlen=3)
    outputs = [corruptor(np.full(5, float(t)), rng) for t in range(1, 6)]
    np.testing.assert_array_equal([o[0] for o in outputs], [0.0, 0.0, 1.0, 2.0, 3.0])


def test_sensor_noise_std() -> None:
    sigma = {"pitch": 0.01, "pitch_rate": 0.05, "yaw_rate": 0.05, "wheel_left": 0.1, "wheel_right": 0.1}
    corruptor = SensorCorruptor({"enabled": True, "max_latency": 0, "noise": sigma})
    rng = np.random.default_rng(0)
    corruptor.reset(rng, np.zeros(5))
    samples = np.array([corruptor(np.zeros(5), rng) for _ in range(20000)])
    np.testing.assert_allclose(samples.std(axis=0), list(sigma.values()), rtol=0.05)


def test_reward_uses_clean_signals() -> None:
    """En el nivel 3 el agente ve lecturas con ruido, pero la recompensa usa el pitch real."""
    env = BalanceBotEnv(level=3)
    env.reset(seed=0)
    diffs = []
    for _ in range(50):
        obs, _, _, _, info = env.step(ZERO)
        assert info["reward_components"]["theta"] == pytest.approx(-info["pitch"] ** 2)
        diffs.append(obs[0] - info["pitch"])
    assert np.std(diffs) > 0.003


# ---------------------------------------------------------------- PID en niveles superiores


@pytest.mark.parametrize("level", [1, 2])
def test_pid_survives_levels_1_2(level: int) -> None:
    env = BalanceBotEnv(level=level)
    results = run_episodes(env, PIDController.from_env(env), seeds=list(range(10)))
    assert not any(r["fell"] for r in results)
