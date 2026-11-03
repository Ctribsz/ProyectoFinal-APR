"""PID en cascada para BalanceBotEnv (entregable E4, especificación §5).

Usa solo la observación del entorno, igual que los agentes. Expone `predict(obs)` como una política de SB3.

Lazo externo (velocidad): θ_ref = Kp_v·e_v + Ki_v·∫e_v, con e_v = v* − v.
    Para acelerar hacia adelante el robot debe inclinarse hacia adelante (θ_ref > 0).
Lazo interno (ángulo):    u = Kp·(θ − θ_ref) + Ki·∫(θ − θ_ref) + Kd·θ̇.
    θ > 0 (cae hacia adelante) pide torque positivo para meter las ruedas debajo (§4.1).
Giro:                     u_izq/der = u ∓ Kp_yaw·(ψ̇* − ψ̇).  ψ̇ > 0 cuando la rueda derecha va más rápido.

Uso:
    python -m baselines.pid_controller --level 0 --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from envs.balance_bot_env import REPO_ROOT, BalanceBotEnv

DEFAULT_CONFIG = REPO_ROOT / "configs" / "baselines" / "pid.yaml"


def load_pid_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


class PIDController:
    def __init__(self, config: dict[str, Any], wheel_radius: float, dt: float) -> None:
        self.wheel_radius = wheel_radius
        self.dt = dt
        inner, outer, yaw = config["inner"], config["outer"], config["yaw"]
        self.kp, self.ki, self.kd = float(inner["kp"]), float(inner["ki"]), float(inner["kd"])
        self.kp_v, self.ki_v = float(outer["kp"]), float(outer["ki"])
        self.max_pitch_ref = float(outer["max_pitch_ref"])
        self.integral_limit = float(config["integral_limit"])
        self.kp_yaw = float(yaw["kp"])
        self.reset()

    @classmethod
    def from_env(cls, env: BalanceBotEnv, config: dict[str, Any] | None = None) -> PIDController:
        """Toma el radio de rueda (MJCF) y el periodo de control del entorno."""
        return cls(config if config is not None else load_pid_config(), env.wheel_radius, env.dt)

    def reset(self) -> None:
        self._int_pitch = 0.0
        self._int_vel = 0.0

    def predict(
        self, observation: np.ndarray, state: Any = None, episode_start: Any = None, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        pitch, pitch_rate, yaw_rate, w_left, w_right, v_cmd, yaw_cmd = (float(x) for x in observation[:7])
        # Misma estimación que el entorno: encoders relativos al chasis + θ̇ (§4.3).
        v = self.wheel_radius * (0.5 * (w_left + w_right) + pitch_rate)

        e_v = v_cmd - v
        self._int_vel = float(np.clip(self._int_vel + e_v * self.dt, -self.integral_limit, self.integral_limit))
        pitch_ref = float(
            np.clip(self.kp_v * e_v + self.ki_v * self._int_vel, -self.max_pitch_ref, self.max_pitch_ref)
        )

        e_pitch = pitch - pitch_ref
        self._int_pitch = float(
            np.clip(self._int_pitch + e_pitch * self.dt, -self.integral_limit, self.integral_limit)
        )
        u = self.kp * e_pitch + self.ki * self._int_pitch + self.kd * pitch_rate

        u_yaw = self.kp_yaw * (yaw_cmd - yaw_rate)
        action = np.clip([u - u_yaw, u + u_yaw], -1.0, 1.0).astype(np.float32)
        return action, None


def run_episodes(env: BalanceBotEnv, controller: PIDController, seeds: list[int]) -> list[dict[str, float]]:
    """Corre un episodio por semilla y devuelve pasos, retorno y si cayó."""
    results = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        controller.reset()
        ret, steps = 0.0, 0
        while True:
            action, _ = controller.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            steps += 1
            if terminated or truncated:
                break
        results.append({"seed": seed, "steps": steps, "return": ret, "fell": terminated})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el PID en cascada")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    env = BalanceBotEnv(level=args.level)
    controller = PIDController.from_env(env, load_pid_config(args.config))
    results = run_episodes(env, controller, list(range(args.episodes)))
    falls = sum(r["fell"] for r in results)
    print(f"Episodios: {len(results)} · caídas: {falls} · retorno medio: {np.mean([r['return'] for r in results]):.1f}")
    env.close()


if __name__ == "__main__":
    main()
