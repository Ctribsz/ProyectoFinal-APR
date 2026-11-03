"""Comandos de velocidad y giro (especificación §4.6, niveles 2–3).

v* ~ U(-v_max, v_max) y ψ̇* ~ U(-yaw_rate_max, yaw_rate_max), que cambian cada U(min_interval, max_interval) s.
Con `enabled: false` los comandos quedan en cero.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CommandGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.v_max = float(config["v_max"])
        self.yaw_rate_max = float(config["yaw_rate_max"])
        self.min_interval = float(config.get("min_interval", 2.0))
        self.max_interval = float(config.get("max_interval", 4.0))
        self.command = np.zeros(2)
        self._next_change = np.inf

    def reset(self, rng: np.random.Generator) -> None:
        self.command = np.zeros(2)
        self._next_change = np.inf
        if self.enabled:
            self._resample(0.0, rng)

    def update(self, t: float, rng: np.random.Generator) -> np.ndarray:
        if t >= self._next_change:
            self._resample(t, rng)
        return self.command

    def set(self, v: float, yaw_rate: float) -> None:
        """Fija un comando constante (para evaluación). Desactiva el cambio aleatorio."""
        self.command = np.array([v, yaw_rate], dtype=np.float64)
        self._next_change = np.inf

    def _resample(self, t: float, rng: np.random.Generator) -> None:
        self.command = np.array(
            [rng.uniform(-self.v_max, self.v_max), rng.uniform(-self.yaw_rate_max, self.yaw_rate_max)]
        )
        self._next_change = t + rng.uniform(self.min_interval, self.max_interval)
