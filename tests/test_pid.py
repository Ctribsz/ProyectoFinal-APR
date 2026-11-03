"""Pruebas del PID en cascada (entregable E4)."""

import numpy as np

from baselines.pid_controller import PIDController, run_episodes
from envs import BalanceBotEnv


def test_pid_balances() -> None:
    """Criterio de salida de F1: el PID sobrevive 1000 pasos en 20 semillas del nivel 0."""
    env = BalanceBotEnv(level=0)
    results = run_episodes(env, PIDController.from_env(env), seeds=list(range(20)))
    assert all(r["steps"] == env.max_episode_steps and not r["fell"] for r in results)


def test_pid_holds_position() -> None:
    """El lazo externo evita que el robot se desplace sin control."""
    env = BalanceBotEnv(level=0)
    run_episodes(env, PIDController.from_env(env), seeds=[0])
    assert abs(env.data.qpos[0]) < 0.05


def test_pid_corrects_in_right_direction() -> None:
    """Inclinado hacia adelante (θ > 0) pide torque positivo; hacia atrás, negativo (§4.1)."""
    env = BalanceBotEnv(level=0)
    pid = PIDController.from_env(env)
    obs = np.zeros(9, dtype=np.float32)
    obs[0] = 0.1
    assert (pid.predict(obs)[0] > 0).all()
    pid.reset()
    obs[0] = -0.1
    assert (pid.predict(obs)[0] < 0).all()


def test_pid_uses_only_observation() -> None:
    """Misma observación y estado interno → misma acción, sin importar el estado del simulador."""
    env = BalanceBotEnv(level=0)
    obs, _ = env.reset(seed=0)
    a1, _ = PIDController.from_env(env).predict(obs)
    env.data.qpos[0] = 5.0  # mover el robot en el simulador no debe afectar
    a2, _ = PIDController.from_env(env).predict(obs)
    np.testing.assert_array_equal(a1, a2)
