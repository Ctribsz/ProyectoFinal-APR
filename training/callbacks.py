"""Callbacks de SB3 para diagnóstico (especificación §8, entregable E9).

- `RewardComponentsCallback`: cada componente de la recompensa (§4.4), en los episodios de entrenamiento.
- `DiagnosticsCallback`: cada `freq` pasos juega episodios deterministas y registra la suavidad de la acción
  (§7.4, los 4 algoritmos) y la sobreestimación de Q (DDPG, TD3, SAC).
- α de SAC (`train/ent_coef`) y KL aproximada / fracción de clip de PPO (`train/approx_kl`,
  `train/clip_fraction`) ya los registra SB3 en TensorBoard; no necesitan callback.

Las funciones `action_smoothness`, `high_freq_energy` y `discounted_returns` son puras para que la evaluación
(E11) calcule las métricas de §7.4 con el mismo código.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from envs.balance_bot_env import BalanceBotEnv


class RewardComponentsCallback(BaseCallback):
    """Registra cada componente de la recompensa por separado (§4.4).

    Por episodio guarda la suma de cada componente crudo (`info["reward_components"]`) y la registra en
    TensorBoard como `reward/<componente>` (media de los episodios terminados desde el último volcado).
    """

    def _on_training_start(self) -> None:
        self._sums: list[dict[str, float]] = [defaultdict(float) for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = self.locals["dones"]
        for i, info in enumerate(infos):
            for key, value in info.get("reward_components", {}).items():
                self._sums[i][key] += value
            if dones[i]:
                for key, total in self._sums[i].items():
                    self.logger.record_mean(f"reward/{key}", float(total))
                self._sums[i] = defaultdict(float)
        return True


def action_smoothness(actions: np.ndarray) -> float:
    """Media de ‖a_t − a_{t−1}‖ a lo largo de un episodio (§7.4). `actions` tiene forma (T, n)."""
    actions = np.asarray(actions, dtype=float)
    if len(actions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(actions, axis=0), axis=1).mean())


def high_freq_energy(actions: np.ndarray, dt: float, cutoff_hz: float) -> float:
    """Fracción de la energía espectral de la acción por encima de `cutoff_hz` (§7.4), entre 0 y 1.

    Se quita la media de cada dimensión (la componente de 0 Hz no cuenta) y se suman las dimensiones.
    """
    actions = np.asarray(actions, dtype=float)
    if len(actions) < 4:
        return 0.0
    spectrum = np.abs(np.fft.rfft(actions - actions.mean(axis=0), axis=0)) ** 2
    freqs = np.fft.rfftfreq(len(actions), d=dt)
    total = spectrum[1:].sum()
    return float(spectrum[freqs > cutoff_hz].sum() / total) if total > 0 else 0.0


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """G_t = Σ_k γ^k r_{t+k} para cada paso de un episodio."""
    returns = np.zeros(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


class DiagnosticsCallback(BaseCallback):
    """Cada `freq` pasos juega `n_episodes` episodios deterministas y registra los diagnósticos de §8.

    Sobreestimación de Q (método del artículo de TD3, ver §8): en cada estado visitado se compara
    Q(s, π(s)) del primer crítico con el retorno descontado que la política obtuvo de verdad desde ese estado en
    el mismo episodio. En episodios cortados por tiempo se descartan los últimos `truncation_margin` pasos,
    porque su retorno real quedó incompleto. En SAC, Q incluye el bono de entropía, así que su sesgo positivo
    es en parte por diseño.

    En TensorBoard: `diagnostics/{q_estimate, q_return, q_overestimation, action_smoothness, high_freq_energy}`.
    En `<log_path>/diagnostics.npz`: las mismas series por diagnóstico, con sus `timesteps`.
    """

    def __init__(self, env_cfg: dict[str, Any], cfg: dict[str, Any], seed: int, log_path: str | Path | None = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.env_cfg = env_cfg
        self.freq = int(cfg["freq"])
        self.n_episodes = int(cfg["n_episodes"])
        self.episode_seed = seed + int(cfg["seed_offset"])
        self.truncation_margin = int(cfg["q_overestimation"]["truncation_margin"])
        self.cutoff_hz = float(cfg["smoothness"]["cutoff_hz"])
        self.log_path = Path(log_path) if log_path is not None else None
        self.history: dict[str, list[float]] = defaultdict(list)

    def _init_callback(self) -> None:
        self.env = BalanceBotEnv(level=self.env_cfg["level"], config=self.env_cfg)
        self._next = self.freq
        self.has_critic = hasattr(self.model, "critic")  # DDPG, TD3 y SAC; PPO solo tiene V(s)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.freq
            self.run_diagnostics()
        return True

    def _on_training_end(self) -> None:
        self.env.close()

    def run_diagnostics(self) -> dict[str, float]:
        vec_normalize = self.model.get_vec_normalize_env()
        smooth, hf, q_est, q_ret = [], [], [], []
        for k in range(self.n_episodes):
            obs, _ = self.env.reset(seed=self.episode_seed + k)
            observations, actions, rewards = [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                policy_obs = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
                action, _ = self.model.predict(policy_obs, deterministic=True)
                observations.append(policy_obs)
                actions.append(action)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(float(reward))

            actions_arr = np.asarray(actions)
            smooth.append(action_smoothness(actions_arr))
            hf.append(high_freq_energy(actions_arr, self.env.dt, self.cutoff_hz))
            if self.has_critic:
                keep = len(rewards) if terminated else max(0, len(rewards) - self.truncation_margin)
                if keep:
                    returns = discounted_returns(np.asarray(rewards), float(self.model.gamma))[:keep]
                    q_est.extend(self.q_values(np.asarray(observations[:keep]), actions_arr[:keep]))
                    q_ret.extend(returns)

        values = {"action_smoothness": float(np.mean(smooth)), "high_freq_energy": float(np.mean(hf))}
        if q_est:
            values |= {"q_estimate": float(np.mean(q_est)), "q_return": float(np.mean(q_ret)), "q_overestimation": float(np.mean(np.subtract(q_est, q_ret)))}
        for key, value in values.items():
            self.logger.record(f"diagnostics/{key}", value)
        self.logger.dump(self.num_timesteps)
        self._save(values)
        return values

    def q_values(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Q(s, a) del primer crítico. El crítico se entrena con acciones escaladas a [−1, 1]."""
        policy = self.model.policy
        obs_tensor, _ = policy.obs_to_tensor(observations)
        act_tensor = torch.as_tensor(policy.scale_action(actions), dtype=torch.float32, device=self.model.device)
        with torch.no_grad():
            return policy.critic(obs_tensor, act_tensor)[0].squeeze(-1).cpu().numpy()

    def _save(self, values: dict[str, float]) -> None:
        self.history["timesteps"].append(self.num_timesteps)
        for key in ("action_smoothness", "high_freq_energy", "q_estimate", "q_return", "q_overestimation"):
            self.history[key].append(values.get(key, np.nan))
        if self.log_path is not None:
            np.savez(self.log_path / "diagnostics.npz", **{k: np.asarray(v) for k, v in self.history.items()})
