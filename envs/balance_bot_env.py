"""Entorno de gymnasium del robot de auto-balanceo (especificación §4).

Las constantes físicas salen del MJCF y el resto de parámetros del YAML del nivel
(`configs/env/level_{n}.yaml`). La convención de ejes está en la especificación §4.1.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from envs.commands import CommandGenerator
from envs.perturbations import PushPerturbation
from envs.randomization import SENSOR_KEYS, DomainRandomizer, SensorCorruptor

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "env"

# Orden de los componentes de la recompensa (§4.4). Cada valor crudo ya lleva su signo,
# así que la recompensa es sum(peso_k * componente_k) con pesos positivos.
REWARD_KEYS = ("alive", "theta", "dtheta", "vel", "yaw", "effort", "smooth")

OBS_DIM = 9


def load_config(level: int) -> dict[str, Any]:
    """Carga `configs/env/level_{level}.yaml`."""
    path = CONFIG_DIR / f"level_{level}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def quat_to_pitch(quat: np.ndarray) -> float:
    """Pitch (rotación alrededor de +y) de un cuaternión (w, x, y, z). θ > 0 es inclinarse hacia adelante."""
    w, x, y, z = quat
    return float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


class BalanceBotEnv(gym.Env):
    """Robot de dos ruedas que debe mantenerse vertical.

    Observación (§4.3), 9 valores: [θ, θ̇, ψ̇, ω_izq, ω_der, v*, ψ̇*, a_izq(t-1), a_der(t-1)].
    Acción (§4.2): comando normalizado en [-1, 1] por motor.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        level: int = 0,
        config: dict[str, Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        if level not in (0, 1, 2, 3):
            raise ValueError(f"Nivel inválido: {level}")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode no soportado: {render_mode}")

        self.level = level
        self.config = config if config is not None else load_config(level)
        if int(self.config["level"]) != level:
            raise ValueError(f"La configuración es del nivel {self.config['level']}, no del {level}")
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(str(REPO_ROOT / self.config["model_path"]))
        self.data = mujoco.MjData(self.model)

        self.frame_skip: int = int(self.config["frame_skip"])
        self.max_episode_steps: int = int(self.config["max_episode_steps"])
        self.termination_pitch: float = float(self.config["termination_pitch"])
        self.initial_pitch_range: float = float(self.config["initial_pitch_range"])
        self.reward_weights: dict[str, float] = {k: float(self.config["reward_weights"][k]) for k in REWARD_KEYS}

        self.wheel_radius: float = float(self.model.geom("wheel_left").size[0])
        self.dt: float = self.model.opt.timestep * self.frame_skip
        self.metadata = {**self.metadata, "render_fps": int(round(1.0 / self.dt))}

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Límites = saturación de los sensores (cutoff del MJCF) y rangos de comando del YAML.
        gyro_max = float(self.model.sensor("imu_gyro").cutoff[0])
        enc_max = float(self.model.sensor("enc_left").cutoff[0])
        commands = self.config["commands"]
        high = np.array(
            [np.pi / 2, gyro_max, gyro_max, enc_max, enc_max, commands["v_max"], commands["yaw_rate_max"], 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._root_qpos = self.model.joint("root").qposadr[0]
        self._prev_action = np.zeros(2, dtype=np.float64)
        self._command = np.zeros(2, dtype=np.float64)  # (v*, ψ̇*); en cero en los niveles 0–1
        self._steps = 0

        self.perturbations = PushPerturbation(self.model, self.config["perturbations"])
        self.commands = CommandGenerator(self.config["commands"])
        self.randomizer = DomainRandomizer(self.model, self.config["randomization"])
        self.sensor_corruptor = SensorCorruptor(self.config["randomization"])

        self._renderer: mujoco.Renderer | None = None
        self._viewer = None  # mujoco.viewer.Handle en modo "human"
        self._last_frame_time: float | None = None

    # ------------------------------------------------------------------ sensores

    def _read_sensors(self) -> dict[str, float]:
        """Lecturas equivalentes a las de una IMU y dos encoders reales."""
        gyro = self.data.sensor("imu_gyro").data
        wheel_left = float(self.data.sensor("enc_left").data[0])
        wheel_right = float(self.data.sensor("enc_right").data[0])
        pitch_rate = float(gyro[1])
        # Los encoders miden la rueda relativa al chasis; se suma θ̇ para la velocidad absoluta.
        v = self.wheel_radius * (0.5 * (wheel_left + wheel_right) + pitch_rate)
        return {
            "pitch": quat_to_pitch(self.data.sensor("imu_quat").data),
            "pitch_rate": pitch_rate,
            "yaw_rate": float(gyro[2]),
            "wheel_left": wheel_left,
            "wheel_right": wheel_right,
            "v": v,
        }

    def _get_obs(self, sensors: np.ndarray) -> np.ndarray:
        """`sensors` son las lecturas que ve el agente (con ruido y latencia en el nivel 3)."""
        obs = np.concatenate([sensors, self._command, self._prev_action]).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ------------------------------------------------------------------ dinámica

    def _apply_action(self, action: np.ndarray) -> None:
        """Acción → control del motor. Torque ideal: el `gear` del MJCF escala [-1, 1] al torque máximo (D1)."""
        self.data.ctrl[:] = action

    def _reward(self, s: dict[str, float], action: np.ndarray) -> tuple[float, dict[str, float]]:
        components = {
            "alive": 1.0,
            "theta": -s["pitch"] ** 2,
            "dtheta": -s["pitch_rate"] ** 2,
            "vel": -(s["v"] - self._command[0]) ** 2,
            "yaw": -(s["yaw_rate"] - self._command[1]) ** 2,
            "effort": -float(np.sum(action**2)),
            "smooth": -float(np.sum((action - self._prev_action) ** 2)),
        }
        reward = sum(self.reward_weights[k] * components[k] for k in REWARD_KEYS)
        return float(reward), components

    # ------------------------------------------------------------------ API gymnasium

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        factors = self.randomizer.reset(self.np_random)

        theta0 = self.np_random.uniform(-self.initial_pitch_range, self.initial_pitch_range)
        q = self._root_qpos
        self.data.qpos[q + 3 : q + 7] = [np.cos(theta0 / 2), 0.0, np.sin(theta0 / 2), 0.0]
        mujoco.mj_forward(self.model, self.data)

        self._prev_action[:] = 0.0
        self._steps = 0
        self.perturbations.reset(self.np_random)
        self.commands.reset(self.np_random)
        self._command[:] = self.commands.command

        s = self._read_sensors()
        clean = np.array([s[k] for k in SENSOR_KEYS])
        self.sensor_corruptor.reset(self.np_random, clean)
        if self.render_mode == "human":
            self.render()
        info = {"pitch": s["pitch"], "randomization": factors, "latency": self.sensor_corruptor.latency}
        return self._get_obs(clean), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._apply_action(action)
        self.perturbations.update(self.data, self.np_random)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._steps += 1

        # Recompensa y terminación con lecturas limpias; el agente ve las corrompidas (nivel 3).
        s = self._read_sensors()
        reward, components = self._reward(s, action)
        self._prev_action[:] = action
        self._command[:] = self.commands.update(self.data.time, self.np_random)
        sensors = self.sensor_corruptor(np.array([s[k] for k in SENSOR_KEYS]), self.np_random)

        terminated = abs(s["pitch"]) > self.termination_pitch
        truncated = (not terminated) and self._steps >= self.max_episode_steps
        info = {
            "reward_components": components,
            "pitch": s["pitch"],
            "v": s["v"],
            "yaw_rate": s["yaw_rate"],
            "command": self._command.copy(),
            "push_active": self.perturbations.active is not None,
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(sensors), reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_rgb()
        if self.render_mode == "human":
            self._render_human()
        return None

    def _render_rgb(self) -> np.ndarray:
        cfg = self.config["render"]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=int(cfg["height"]), width=int(cfg["width"]))
        self._renderer.update_scene(self.data, camera=cfg["camera"])
        return self._renderer.render()

    def _render_human(self) -> None:
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._viewer.cam.fixedcamid = self.model.camera(self.config["render"]["camera"]).id
        if not self._viewer.is_running():
            return
        self._viewer.sync()
        # Ritmo de tiempo real: un paso de control cada dt segundos.
        now = time.perf_counter()
        if self._last_frame_time is not None:
            time.sleep(max(0.0, self.dt - (now - self._last_frame_time)))
        self._last_frame_time = time.perf_counter()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
