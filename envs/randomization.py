"""Aleatorización del nivel 3 (especificación §4.6).

- `DomainRandomizer`: por episodio escala la masa del chasis, la fricción rueda-suelo y el torque máximo,
  siempre a partir de los valores nominales del MJCF.
- `SensorCorruptor`: ruido gaussiano y latencia (0–N pasos, fija por episodio) en las lecturas de los sensores
  (observaciones 0–4). La recompensa y la terminación usan las lecturas limpias.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import mujoco
import numpy as np

SENSOR_KEYS = ("pitch", "pitch_rate", "yaw_rate", "wheel_left", "wheel_right")


class DomainRandomizer:
    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.model = model
        self.mass_range = float(config.get("mass_range", 0.0))
        self.friction_range = float(config.get("friction_range", 0.0))
        self.torque_range = float(config.get("torque_range", 0.0))

        self._chassis = model.body("chassis").id
        # La fricción de un contacto es el máximo de las dos geometrías: se escalan ruedas y suelo juntos.
        self._friction_geoms = [model.geom(n).id for n in ("wheel_left", "wheel_right", "floor")]
        self._nominal_mass = model.body_mass[self._chassis].copy()
        self._nominal_inertia = model.body_inertia[self._chassis].copy()
        self._nominal_friction = model.geom_friction[self._friction_geoms, 0].copy()
        self._nominal_gear = model.actuator_gear[:, 0].copy()
        self.factors = {"mass": 1.0, "friction": 1.0, "torque": 1.0}

    def reset(self, rng: np.random.Generator) -> dict[str, float]:
        if self.enabled:
            self.factors = {
                "mass": 1.0 + rng.uniform(-self.mass_range, self.mass_range),
                "friction": 1.0 + rng.uniform(-self.friction_range, self.friction_range),
                "torque": 1.0 + rng.uniform(-self.torque_range, self.torque_range),
            }
        else:
            self.factors = {"mass": 1.0, "friction": 1.0, "torque": 1.0}
        m = self.model
        m.body_mass[self._chassis] = self._nominal_mass * self.factors["mass"]
        m.body_inertia[self._chassis] = self._nominal_inertia * self.factors["mass"]  # misma geometría
        m.geom_friction[self._friction_geoms, 0] = self._nominal_friction * self.factors["friction"]
        m.actuator_gear[:, 0] = self._nominal_gear * self.factors["torque"]
        return dict(self.factors)


class SensorCorruptor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        noise = config.get("noise", {}) or {}
        self.sigma = np.array([float(noise.get(k, 0.0)) for k in SENSOR_KEYS])
        self.max_latency = int(config.get("max_latency", 0))
        self.latency = 0
        self._buffer: deque[np.ndarray] = deque(maxlen=1)

    def reset(self, rng: np.random.Generator, clean: np.ndarray) -> None:
        self.latency = int(rng.integers(0, self.max_latency + 1)) if self.enabled else 0
        self._buffer = deque([clean.copy()] * (self.latency + 1), maxlen=self.latency + 1)

    def __call__(self, clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Recibe las lecturas limpias del paso actual y devuelve las que ve el agente."""
        if not self.enabled:
            return clean
        self._buffer.append(clean.copy())
        delayed = self._buffer[0]  # lectura de hace `latency` pasos
        return delayed + rng.normal(0.0, 1.0, size=delayed.shape) * self.sigma
