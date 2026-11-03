"""Callbacks de SB3 para diagnóstico (especificación §8). Los diagnósticos completos se agregan en E9."""

from __future__ import annotations

from collections import defaultdict

from stable_baselines3.common.callbacks import BaseCallback


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
