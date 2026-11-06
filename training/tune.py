"""Afinamiento de hiperparámetros con Optuna (especificación §6, entregable E9).

Uso:
    python -m training.tune --algo ppo [--env-config configs/env/level_2.yaml] [--n-trials 40] [--trial-timesteps 200000]

Cada prueba muestrea `search_space` de `configs/algos/<algo>.yaml` y entrena con el mismo `train()` que las
corridas finales (regla de un solo pipeline, AGENTS.md). El protocolo (pasos, semillas, métrica, muestreador y
podador) sale de `configs/training/tune.yaml`.

Resultados en `<storage_dir>/`: el estudio en `<algo>_level<n>.db` y cada corrida completa (config.yaml,
meta.json, evaluations.npz, model.zip, TensorBoard) en `<algo>_level<n>/trial<k>_seed<s>/`.

Paralelismo: varios procesos con el mismo comando comparten el estudio a través del `.db`. Cada proceso lleva
un `--worker` distinto para que sus muestreadores no propongan las mismas configuraciones.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from stable_baselines3.common.callbacks import BaseCallback

from envs.balance_bot_env import REPO_ROOT
from training.train import ALGOS, COMMON_CONFIG, load_yaml, train

TUNE_CONFIG = REPO_ROOT / "configs" / "training" / "tune.yaml"


def sample_params(trial: optuna.Trial, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Muestrea un valor por entrada de `search_space`. Las claves se conservan tal cual (p. ej. `action_noise.sigma`)."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "float":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Tipo de búsqueda no soportado en '{name}': {kind}")
    return params


def apply_params(algo_cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Copia de `algo_cfg` con los valores muestreados en `hyperparams`; `a.b` se escribe en `hyperparams.a.b`."""
    cfg = copy.deepcopy(algo_cfg)
    hyperparams = cfg.setdefault("hyperparams", {})
    for name, value in params.items():
        *parents, leaf = name.split(".")
        node = hyperparams
        for key in parents:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"'{name}': hyperparams.{key} no es un diccionario")
        node[leaf] = value
    return cfg


def objective_value(results: Any, last_k: int) -> float:
    """Media de las últimas `last_k` evaluaciones; `results` tiene forma (n_evaluaciones, n_episodios)."""
    per_eval = np.asarray(results, dtype=float).mean(axis=1)
    return float(per_eval[-last_k:].mean())


class TrialPruningCallback(BaseCallback):
    """Reporta a Optuna después de cada evaluación y detiene la corrida si el podador lo indica.

    Se pasa como `callback_after_eval` de `EvalCallback`, así que `self.parent` es ese `EvalCallback`. Al
    podar devuelve False: `train()` termina normalmente y la corrida queda guardada completa.
    """

    def __init__(self, trial: optuna.Trial, last_k: int, step_offset: int = 0) -> None:
        super().__init__()
        self.trial = trial
        self.last_k = last_k
        self.step_offset = step_offset  # con varias semillas por prueba, cada semilla reporta en su propio tramo
        self.pruned = False

    def _on_step(self) -> bool:
        value = objective_value(self.parent.evaluations_results, self.last_k)
        self.trial.report(value, self.step_offset + self.num_timesteps)
        if self.trial.should_prune():
            self.pruned = True
            return False
        return True


def make_sampler(cfg: dict[str, Any], worker: int = 0) -> optuna.samplers.BaseSampler:
    if cfg["type"] != "tpe":
        raise ValueError(f"Muestreador no soportado: {cfg['type']}")
    return optuna.samplers.TPESampler(seed=int(cfg["seed"]) + worker, n_startup_trials=int(cfg["n_startup_trials"]))


def make_pruner(cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    if cfg["type"] != "median":
        raise ValueError(f"Podador no soportado: {cfg['type']}")
    return optuna.pruners.MedianPruner(n_startup_trials=int(cfg["n_startup_trials"]), n_warmup_steps=int(cfg["n_warmup_steps"]))


def run_study(
    algo: str,
    env_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
    common_cfg: dict[str, Any],
    tune_cfg: dict[str, Any],
    n_trials: int | None = None,
    trial_timesteps: int | None = None,
    storage_dir: str | Path | None = None,
    worker: int = 0,
) -> optuna.Study:
    """Crea o continúa el estudio `<algo>_level<n>` y corre `n_trials` pruebas más en este proceso."""
    if algo not in ALGOS:
        raise ValueError(f"Algoritmo desconocido: {algo}")
    if algo_cfg.get("algo") != algo:
        raise ValueError(f"La configuración es de '{algo_cfg.get('algo')}', no de '{algo}'")
    search_space = algo_cfg.get("search_space") or {}
    if not search_space:
        raise ValueError(f"configs/algos/{algo}.yaml no tiene search_space")

    n_trials = int(n_trials if n_trials is not None else tune_cfg["n_trials"])
    trial_timesteps = int(trial_timesteps if trial_timesteps is not None else tune_cfg["trial_timesteps"])
    seeds = [int(s) for s in tune_cfg["seeds"]]
    last_k = int(tune_cfg["objective"]["last_k_evals"])

    storage_dir = REPO_ROOT / (storage_dir if storage_dir is not None else tune_cfg["storage_dir"])
    study_name = f"{algo}_level{env_cfg['level']}"
    runs_root = storage_dir / study_name
    runs_root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{(storage_dir / f'{study_name}.db').as_posix()}",
        direction="maximize",
        sampler=make_sampler(tune_cfg["sampler"], worker),
        pruner=make_pruner(tune_cfg["pruner"]),
        load_if_exists=True,
    )

    # Todas las pruebas de un estudio deben compartir protocolo; si no, sus valores no son comparables.
    protocol = {
        "trial_timesteps": trial_timesteps,
        "seeds": seeds,
        "objective": tune_cfg["objective"],
        "env": env_cfg,
        "common": common_cfg,
    }
    stored = study.user_attrs.get("protocol")
    if stored is None:
        study.set_user_attr("protocol", protocol)
    elif stored != protocol:
        raise ValueError(f"El estudio '{study_name}' ya existe con otro protocolo; usa otro --storage-dir")

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = apply_params(algo_cfg, sample_params(trial, search_space))
        values, run_dirs = [], []
        for i, seed in enumerate(seeds):
            pruning = TrialPruningCallback(trial, last_k, step_offset=i * trial_timesteps)
            run_dir = train(
                algo,
                env_cfg,
                trial_cfg,
                common_cfg,
                seed=seed,
                timesteps=trial_timesteps,
                out_root=runs_root,
                run_name=f"trial{trial.number:03d}_seed{seed}",
                callback_after_eval=pruning,
            )
            run_dirs.append(str(run_dir))
            trial.set_user_attr("run_dirs", run_dirs)
            if pruning.pruned:
                raise optuna.TrialPruned()
            values.append(objective_value(np.load(run_dir / "evaluations.npz")["results"], last_k))
        return float(np.mean(values))

    # ValueError: SB3 lo lanza cuando la política diverge a NaN con hiperparámetros extremos. La prueba queda
    # como FAIL en el estudio y el resto sigue.
    study.optimize(objective, n_trials=n_trials, catch=(ValueError,))
    return study


def main() -> None:
    parser = argparse.ArgumentParser(description="Afina los hiperparámetros de un algoritmo con Optuna")
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--env-config", default=None, help="por defecto env_config de tune.yaml (nivel 2)")
    parser.add_argument("--algo-config", default=None, help="por defecto configs/algos/<algo>.yaml")
    parser.add_argument("--common-config", default=str(COMMON_CONFIG))
    parser.add_argument("--tune-config", default=str(TUNE_CONFIG))
    parser.add_argument("--n-trials", type=int, default=None, help="pruebas que corre este proceso")
    parser.add_argument("--trial-timesteps", type=int, default=None, help="sobrescribe trial_timesteps de tune.yaml")
    parser.add_argument("--storage-dir", default=None, help="sobrescribe storage_dir de tune.yaml")
    parser.add_argument("--worker", type=int, default=0, help="índice del proceso si varios comparten el estudio")
    args = parser.parse_args()

    tune_cfg = load_yaml(args.tune_config)
    env_config = args.env_config or REPO_ROOT / tune_cfg["env_config"]
    algo_config = args.algo_config or REPO_ROOT / "configs" / "algos" / f"{args.algo}.yaml"
    study = run_study(
        args.algo,
        load_yaml(env_config),
        load_yaml(algo_config),
        load_yaml(args.common_config),
        tune_cfg,
        n_trials=args.n_trials,
        trial_timesteps=args.trial_timesteps,
        storage_dir=args.storage_dir,
        worker=args.worker,
    )

    states = [t.state for t in study.trials]
    count = {s.name: states.count(s) for s in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED, optuna.trial.TrialState.FAIL)}
    print(f"{study.study_name}: {len(states)} pruebas ({count['COMPLETE']} completas, {count['PRUNED']} podadas, {count['FAIL']} fallidas)")
    if count["COMPLETE"]:
        print(f"  mejor prueba: #{study.best_trial.number}, objetivo {study.best_value:.1f}")
        for name, value in study.best_params.items():
            print(f"    {name}: {value}")


if __name__ == "__main__":
    main()
