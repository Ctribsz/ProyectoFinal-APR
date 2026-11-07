"""Pipeline común de entrenamiento para DDPG, TD3, SAC y PPO (entregable E6).

Uso:
    python -m training.train --algo sac --env-config configs/env/level_0.yaml --seed 0 [--timesteps 100000]

Todos los algoritmos pasan por este mismo script; solo cambia `configs/algos/<algo>.yaml` (AGENTS.md).
Cada corrida deja en `results/<id>/`: config.yaml, meta.json, evaluations.npz, model.zip y logs de TensorBoard.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from envs.balance_bot_env import REPO_ROOT, BalanceBotEnv
from training.callbacks import DiagnosticsCallback, RewardComponentsCallback

ALGOS: dict[str, type[BaseAlgorithm]] = {"ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
COMMON_CONFIG = REPO_ROOT / "configs" / "training" / "common.yaml"
DIAGNOSTICS_CONFIG = REPO_ROOT / "configs" / "training" / "diagnostics.yaml"
LIBRARIES = ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy", "pyyaml")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def build_hyperparams(algo_cfg: dict[str, Any], n_actions: int) -> dict[str, Any]:
    """Convierte los hiperparámetros del YAML a argumentos de SB3 (p. ej. `action_noise` → objeto de SB3)."""
    params = copy.deepcopy(algo_cfg.get("hyperparams") or {})
    noise = params.pop("action_noise", None)
    if noise is not None:
        if noise["type"] != "normal":
            raise ValueError(f"Tipo de ruido no soportado: {noise['type']}")
        params["action_noise"] = NormalActionNoise(np.zeros(n_actions), float(noise["sigma"]) * np.ones(n_actions))
    return params


def make_env(env_cfg: dict[str, Any], n_envs: int, seed: int) -> VecEnv:
    """VecEnv con Monitor; cada sub-entorno recibe la semilla `seed + i`."""
    return make_vec_env(BalanceBotEnv, n_envs=n_envs, seed=seed, env_kwargs={"level": env_cfg["level"], "config": env_cfg})


def library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for lib in LIBRARIES:
        try:
            versions[lib] = version(lib)
        except Exception:
            versions[lib] = "desconocida"
    return versions


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def train(
    algo: str,
    env_cfg: dict[str, Any],
    algo_cfg: dict[str, Any],
    common_cfg: dict[str, Any],
    seed: int,
    timesteps: int | None = None,
    out_root: str | Path = REPO_ROOT / "results",
    run_name: str | None = None,
    callback_after_eval: BaseCallback | None = None,
    extra_callbacks: list[BaseCallback] | None = None,
    diagnostics_cfg: dict[str, Any] | None = None,
) -> Path:
    """Entrena una corrida y devuelve el directorio de resultados.

    `callback_after_eval` se llama después de cada evaluación (p. ej. la poda de `tune.py`); `extra_callbacks`
    se agregan a los de siempre. Con `diagnostics_cfg` (configs/training/diagnostics.yaml) se registran los
    diagnósticos de §8; el afinamiento no los usa.
    """
    if algo not in ALGOS:
        raise ValueError(f"Algoritmo desconocido: {algo}")
    if algo_cfg.get("algo") != algo:
        raise ValueError(f"La configuración es de '{algo_cfg.get('algo')}', no de '{algo}'")
    total_timesteps = int(timesteps if timesteps is not None else common_cfg["total_timesteps"])
    torch.set_num_threads(int(common_cfg["torch_threads"]))

    run_name = run_name or f"{algo}_level{env_cfg['level']}_seed{seed}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {"algo": algo_cfg, "env": env_cfg, "common": common_cfg, "seed": seed, "total_timesteps": total_timesteps}
    if diagnostics_cfg is not None:
        config["diagnostics"] = diagnostics_cfg
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    n_envs = int(algo_cfg.get("n_envs", 1))
    eval_cfg = common_cfg["eval"]
    train_env = make_env(env_cfg, n_envs, seed)
    eval_env = make_env(env_cfg, 1, seed + int(eval_cfg["seed_offset"]))
    if common_cfg["normalize"]:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = ALGOS[algo](
        common_cfg["policy"],
        train_env,
        seed=seed,
        device=common_cfg["device"],
        policy_kwargs={"net_arch": list(common_cfg["net_arch"])},
        tensorboard_log=str(run_dir / "tb"),
        verbose=0,
        **build_hyperparams(algo_cfg, train_env.action_space.shape[0]),
    )

    callbacks = CallbackList(
        [
            RewardComponentsCallback(),
            EvalCallback(
                eval_env,
                callback_after_eval=callback_after_eval,
                eval_freq=max(int(eval_cfg["freq"]) // n_envs, 1),
                n_eval_episodes=int(eval_cfg["n_episodes"]),
                deterministic=True,
                log_path=str(run_dir),
                verbose=0,
            ),
            *([DiagnosticsCallback(env_cfg, diagnostics_cfg, seed, log_path=run_dir)] if diagnostics_cfg is not None else []),
            *(extra_callbacks or []),
        ]
    )

    started = datetime.now()
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name="run", progress_bar=False)
    duration = time.perf_counter() - t0

    model.save(run_dir / "model.zip")
    if common_cfg["normalize"]:
        train_env.save(str(run_dir / "vecnormalize.pkl"))

    meta = {
        "algo": algo,
        "seed": seed,
        "eval_seed": seed + int(eval_cfg["seed_offset"]),
        "total_timesteps": total_timesteps,
        "timesteps_done": model.num_timesteps,  # menos si un callback detuvo la corrida; PPO redondea a rollouts completos
        "started": started.isoformat(timespec="seconds"),
        "duration_s": round(duration, 2),
        "steps_per_s": round(model.num_timesteps / duration, 1),
        "versions": library_versions(),
        "git": git_state(),
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    with (run_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    train_env.close()
    eval_env.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un agente de SB3 en BalanceBotEnv")
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--env-config", default=str(REPO_ROOT / "configs" / "env" / "level_0.yaml"))
    parser.add_argument("--algo-config", default=None, help="por defecto configs/algos/<algo>.yaml")
    parser.add_argument("--common-config", default=str(COMMON_CONFIG))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=None, help="sobrescribe total_timesteps de common.yaml")
    parser.add_argument("--out", default=str(REPO_ROOT / "results"))
    parser.add_argument("--diagnostics-config", default=str(DIAGNOSTICS_CONFIG), help="diagnósticos de §8")
    parser.add_argument("--no-diagnostics", action="store_true", help="sin diagnósticos de §8 (más rápido)")
    args = parser.parse_args()

    algo_config = args.algo_config or REPO_ROOT / "configs" / "algos" / f"{args.algo}.yaml"
    run_dir = train(
        args.algo,
        load_yaml(args.env_config),
        load_yaml(algo_config),
        load_yaml(args.common_config),
        seed=args.seed,
        timesteps=args.timesteps,
        out_root=args.out,
        diagnostics_cfg=None if args.no_diagnostics else load_yaml(args.diagnostics_config),
    )
    meta = json.loads((run_dir / "meta.json").read_text())
    evals = np.load(run_dir / "evaluations.npz")
    print(f"{run_dir}")
    print(f"  {meta['total_timesteps']} pasos en {meta['duration_s']:.0f} s ({meta['steps_per_s']:.0f} pasos/s)")
    print(f"  retorno de evaluación: primero {evals['results'][0].mean():.1f} → último {evals['results'][-1].mean():.1f}")


if __name__ == "__main__":
    main()
