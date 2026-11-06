"""Mide cuántos pasos/s rinde esta computadora para el afinamiento (E10).

Uso:
    python -m lab.benchmark [--algos ddpg ppo] [--parallel 4] [--quick]

1. Entrena cada algoritmo en un solo proceso, con `train()` y los YAML del repo, en el nivel de `tune.yaml`.
2. Corre `--parallel` procesos de DDPG a la vez (el algoritmo más lento) y mide el rendimiento total.
3. Estima las horas de un estudio completo de `tune.yaml` sin poda.

Las corridas se guardan en una carpeta temporal y se borran al terminar.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from envs.balance_bot_env import REPO_ROOT
from training.train import COMMON_CONFIG, load_yaml, train
from training.tune import TUNE_CONFIG

# Pasos de la medición (no son hiperparámetros): lo justo para que el tiempo de arranque no pese.
BENCH_STEPS = {"ddpg": 10000, "td3": 10000, "sac": 10000, "ppo": 50000}


def run_one(algo: str, timesteps: int, out_root: str, run_name: str) -> dict:
    tune_cfg = load_yaml(TUNE_CONFIG)
    run_dir = train(
        algo,
        load_yaml(REPO_ROOT / tune_cfg["env_config"]),
        load_yaml(REPO_ROOT / "configs" / "algos" / f"{algo}.yaml"),
        load_yaml(COMMON_CONFIG),
        seed=0,
        timesteps=timesteps,
        out_root=out_root,
        run_name=run_name,
    )
    return json.loads((run_dir / "meta.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide los pasos/s de esta computadora")
    parser.add_argument("--algos", nargs="+", default=["ddpg", "ppo"], choices=sorted(BENCH_STEPS))
    parser.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="procesos de DDPG a la vez")
    parser.add_argument("--quick", action="store_true", help="10 veces menos pasos (solo para probar que funciona)")
    args = parser.parse_args()

    scale = 10 if args.quick else 1
    tune_cfg = load_yaml(TUNE_CONFIG)
    level = load_yaml(REPO_ROOT / tune_cfg["env_config"])["level"]
    study_steps = int(tune_cfg["n_trials"]) * int(tune_cfg["trial_timesteps"])
    print(f"Computadora: {os.cpu_count()} hilos lógicos. Nivel {level}. Esto tarda unos 10 minutos; no uses la PC mientras.")

    with tempfile.TemporaryDirectory() as tmp:
        print("\n1) Un solo proceso")
        for algo in args.algos:
            meta = run_one(algo, BENCH_STEPS[algo] // scale, tmp, f"{algo}_single")
            hours = study_steps / meta["steps_per_s"] / 3600
            print(f"   {algo:5s} {meta['steps_per_s']:7.0f} pasos/s   estudio completo en 1 proceso, sin poda: ~{hours:.0f} h", flush=True)

        n = args.parallel
        steps = BENCH_STEPS["ddpg"] // scale
        print(f"\n2) {n} procesos de DDPG a la vez")
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n) as pool:
            metas = list(pool.map(run_one, ["ddpg"] * n, [steps] * n, [tmp] * n, [f"ddpg_par{i}" for i in range(n)]))
        wall = time.perf_counter() - t0
        total = sum(m["timesteps_done"] for m in metas) / wall
        per_proc = sum(m["steps_per_s"] for m in metas) / n
        print(f"   {per_proc:.0f} pasos/s por proceso, {total:.0f} pasos/s en total")
        print(f"   estudio completo de DDPG con {n} procesos, sin poda: ~{study_steps / total / 3600:.1f} h (con poda, menos)")

    print("\nCompara 'en total' con distintos --parallel: el mejor número de procesos es donde deja de subir.")


if __name__ == "__main__":
    main()
