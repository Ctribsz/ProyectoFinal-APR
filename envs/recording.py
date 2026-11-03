"""Grabación de episodios a .mp4 (entregable E3).

Uso:
    python -m envs.recording --policy zero --level 0 --seed 0 --out results/videos/caida.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol

import imageio.v2 as imageio
import numpy as np

from envs.balance_bot_env import BalanceBotEnv


class Policy(Protocol):
    """Interfaz común a SB3, el PID y cualquier política: `predict(obs) -> (acción, estado)`."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, Any]: ...


class ZeroPolicy:
    """Política nula: motores apagados. Sirve para ver la caída libre."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(2, dtype=np.float32), None


def record_episode(env: BalanceBotEnv, policy: Policy, path: str | Path, seed: int) -> dict[str, Any]:
    """Corre un episodio con `policy` y lo guarda en `path`. Devuelve un resumen del episodio."""
    if env.render_mode != "rgb_array":
        raise ValueError("El entorno debe crearse con render_mode='rgb_array'.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    episode_return, steps, terminated = 0.0, 0, False
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        episode_return += reward
        steps += 1
        if terminated or truncated:
            break

    imageio.mimsave(path, frames, fps=env.metadata["render_fps"], macro_block_size=1)
    return {"path": str(path), "steps": steps, "return": episode_return, "terminated": terminated}


def _make_policy(name: str, env: BalanceBotEnv) -> Policy:
    if name == "zero":
        return ZeroPolicy()
    if name == "pid":
        from baselines.pid_controller import PIDController

        return PIDController.from_env(env)
    raise ValueError(f"Política desconocida: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Graba un episodio de BalanceBotEnv a .mp4")
    parser.add_argument("--policy", default="zero", help="zero | pid (modelos de SB3 en E6)")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/videos/episodio.mp4")
    args = parser.parse_args()

    env = BalanceBotEnv(level=args.level, render_mode="rgb_array")
    try:
        summary = record_episode(env, _make_policy(args.policy, env), args.out, seed=args.seed)
    finally:
        env.close()
    print(summary)


if __name__ == "__main__":
    main()
