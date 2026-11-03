"""Empujones externos sobre el chasis (especificación §4.6, niveles 1–3).

Fuerza horizontal aplicada en el site `push_point` del chasis durante `duration` segundos. Se implementa con
`data.xfrc_applied` (fuerza en el centro de masa más el torque r × F equivalente).
Los instantes siguen un proceso de Poisson con intervalo medio `mean_interval`; la magnitud ~ U(0, F_max).
Dirección (`direction`):
  - `sagittal`: hacia adelante o atrás del robot (±x del chasis), la que se puede corregir balanceando;
  - `any`: ángulo ~ U(0, 2π) en el plano horizontal, relativo al frente del robot. Un empujón lateral vuelca
    el robot sobre el eje de las ruedas y ningún controlador puede corregirlo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


@dataclass
class Push:
    start: float      # s, tiempo de simulación
    force: float      # N
    angle: float      # rad, relativo al frente del robot en el plano horizontal (0 = adelante, π = atrás)
    duration: float   # s


class PushPerturbation:
    def __init__(self, model: mujoco.MjModel, config: dict[str, Any]) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.body_id = model.body("chassis").id
        self.site_id = model.site("push_point").id
        self.direction = str(config.get("direction", "sagittal"))
        if self.direction not in ("sagittal", "any"):
            raise ValueError(f"direction inválida: {self.direction}")
        self.f_max = float(config.get("f_max", 0.0))
        self.duration = float(config.get("duration", 0.1))
        self.mean_interval = float(config.get("mean_interval", 2.0))
        self._scheduled: list[Push] = []
        self._next_random: float = np.inf
        self.active: Push | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._scheduled = []
        self.active = None
        self._next_random = rng.exponential(self.mean_interval) if self.enabled else np.inf

    def schedule(self, start: float, force: float, angle: float = 0.0, duration: float | None = None) -> None:
        """Programa un empujón exacto (para la prueba de empujón, §7.4). Funciona aunque `enabled` sea False."""
        self._scheduled.append(Push(start, force, angle, self.duration if duration is None else duration))

    def update(self, data: mujoco.MjData, rng: np.random.Generator) -> None:
        """Llamar antes de cada paso de control: activa, mantiene o retira la fuerza."""
        t = data.time
        if self.active is not None and t >= self.active.start + self.active.duration:
            self.active = None
        if self.active is None:
            due = [p for p in self._scheduled if p.start <= t]
            if due:
                self.active = due[0]
                self._scheduled.remove(due[0])
            elif t >= self._next_random:
                angle = rng.choice([0.0, np.pi]) if self.direction == "sagittal" else rng.uniform(0.0, 2 * np.pi)
                self.active = Push(t, rng.uniform(0.0, self.f_max), float(angle), self.duration)
                self._next_random = t + self.duration + rng.exponential(self.mean_interval)

        wrench = np.zeros(6)
        if self.active is not None:
            # Frente del robot proyectado al plano horizontal (columna x de la orientación del chasis).
            fwd = data.xmat[self.body_id].reshape(3, 3)[:, 0].copy()
            fwd[2] = 0.0
            fwd /= max(np.linalg.norm(fwd), 1e-9)
            left = np.array([-fwd[1], fwd[0], 0.0])
            f = self.active.force * (np.cos(self.active.angle) * fwd + np.sin(self.active.angle) * left)
            r = data.site_xpos[self.site_id] - data.xipos[self.body_id]  # del centro de masa al punto de empuje
            wrench[:3] = f
            wrench[3:] = np.cross(r, f)
        data.xfrc_applied[self.body_id] = wrench
