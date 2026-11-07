"""Corre las corridas finales (E12, especificación §7.1) de una lista de semillas, varias a la vez.

Uso:
    python -m lab.semillas --algo ddpg --level 2 --seeds 0-3 7 [--workers 3] [--out results/final]

- Cada semilla es una corrida de `train()` con `configs/algos/<algo>.yaml`, `configs/env/level_<n>.yaml` y
  `common.yaml`, en `<out>/<algo>_level<n>_seed<s>/`. El nombre es fijo: así se juntan las corridas de varias
  máquinas en una sola carpeta para el análisis.
- Se puede volver a lanzar con el mismo comando: las corridas terminadas (tienen `meta.json`, que `train()`
  escribe al final) se saltan. Una carpeta sin `meta.json` es una corrida cortada: se mueve a
  `<out>/_incompletas/` y la semilla se repite desde cero (`train()` no guarda puntos de control).
- Mientras una corrida corre, su `<out>/<nombre>.lock` se toca cada pocos segundos; otro lanzador en esta
  máquina no la toca. Log de cada corrida en `<out>/logs/<nombre>.log`.
- Reutiliza `train()`: no hay un pipeline aparte (AGENTS.md).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from envs.balance_bot_env import REPO_ROOT
from training.train import ALGOS, COMMON_CONFIG, load_yaml, train

from lab.afinar import HEARTBEAT_S, keep_awake

DEFAULT_OUT = REPO_ROOT / "results" / "final"
INCOMPLETE_DIR = "_incompletas"
STATUS_EVERY_S = 60


def parse_seeds(tokens: list[str]) -> list[int]:
    """`["0-3", "7"]` → `[0, 1, 2, 3, 7]`, sin repetidos y en orden."""
    seeds: list[int] = []
    for token in tokens:
        low, _, high = token.partition("-")
        for s in range(int(low), int(high or low) + 1):
            if s not in seeds:
                seeds.append(s)
    return seeds


def run_name(algo: str, level: int, seed: int) -> str:
    return f"{algo}_level{level}_seed{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "meta.json").exists()


def lock_alive(lock: Path) -> bool:
    return lock.exists() and time.time() - lock.stat().st_mtime < HEARTBEAT_S


def plan(out: Path, algo: str, level: int, seeds: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Separa las semillas en (pendientes, terminadas, corriendo en otro lanzador de esta máquina)."""
    pending, done, busy = [], [], []
    for seed in seeds:
        name = run_name(algo, level, seed)
        if is_done(out / name):
            done.append(seed)
        elif lock_alive(out / f"{name}.lock"):
            busy.append(seed)
        else:
            pending.append(seed)
    return pending, done, busy


def set_aside(run_dir: Path, out: Path) -> Path:
    """Mueve una corrida cortada a `<out>/_incompletas/` (no se borra: queda para revisar)."""
    target = out / INCOMPLETE_DIR / f"{run_dir.name}_{datetime.now():%Y%m%d-%H%M%S}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    return target


def run_one(algo: str, level: int, seed: int, out: Path, timesteps: int | None) -> Path:
    """Una corrida final; es lo que ejecuta cada proceso hijo.

    El hijo también toca su .lock: si el lanzador muere y el hijo sigue vivo, otro lanzador no repite la corrida.
    """
    name = run_name(algo, level, seed)
    lock = out / f"{name}.lock"
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(HEARTBEAT_S / 10):
            lock.touch()

    lock.touch()
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        env_cfg = load_yaml(REPO_ROOT / "configs" / "env" / f"level_{level}.yaml")
        algo_cfg = load_yaml(REPO_ROOT / "configs" / "algos" / f"{algo}.yaml")
        return train(algo, env_cfg, algo_cfg, load_yaml(COMMON_CONFIG), seed=seed, timesteps=timesteps, out_root=out, run_name=name)
    finally:
        stop.set()
        lock.unlink(missing_ok=True)


def child_command(algo: str, level: int, seed: int, out: Path, timesteps: int | None) -> list[str]:
    cmd = [sys.executable, "-m", "lab.semillas", "--algo", algo, "--level", str(level), "--seeds", str(seed), "--out", str(out), "--run-one"]
    return cmd + (["--timesteps", str(timesteps)] if timesteps else [])


def launch(algo: str, level: int, seeds: list[int], workers: int, out: Path, timesteps: int | None) -> int:
    """Corre las semillas pendientes con hasta `workers` procesos. Devuelve cuántas fallaron."""
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)
    pending, done, busy = plan(out, algo, level, seeds)
    print(f"{algo}, nivel {level}: {len(done)} terminadas {done}, {len(pending)} por correr {pending}"
          + (f", {len(busy)} ya corriendo en otro lanzador {busy}" if busy else ""), flush=True)
    if not pending:
        return 0

    env = {**os.environ, "PYTHONUTF8": "1"}
    running: dict[int, tuple[subprocess.Popen, object, Path]] = {}
    failed: list[int] = []
    finished = 0
    last = time.monotonic()
    keep_awake(True)
    try:
        while pending or running:
            while pending and len(running) < workers:
                seed = pending.pop(0)
                name = run_name(algo, level, seed)
                if is_done(out / name) or lock_alive(out / f"{name}.lock"):  # la tomó otro lanzador mientras tanto
                    continue
                if (out / name).exists():
                    print(f"  {name}: quedó a medias; se mueve a {set_aside(out / name, out)} y se repite", flush=True)
                lock = out / f"{name}.lock"
                lock.touch()
                log = (log_dir / f"{name}.log").open("a", encoding="utf-8")
                proc = subprocess.Popen(child_command(algo, level, seed, out, timesteps), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                running[seed] = (proc, log, lock)
                print(f"[{datetime.now():%H:%M}] empieza {name}", flush=True)

            for seed, (proc, log, lock) in list(running.items()):
                lock.touch()
                if proc.poll() is None:
                    continue
                log.close()
                lock.unlink(missing_ok=True)
                del running[seed]
                name = run_name(algo, level, seed)
                if proc.returncode == 0 and is_done(out / name):
                    finished += 1
                    print(f"[{datetime.now():%H:%M}] terminó {name}", flush=True)
                else:
                    failed.append(seed)
                    print(f"[{datetime.now():%H:%M}] ERROR en {name} (código {proc.returncode}); ver {log_dir / (name + '.log')}", flush=True)

            if time.monotonic() - last >= STATUS_EVERY_S:
                names = [run_name(algo, level, s) for s in running]
                print(f"[{datetime.now():%H:%M}] {finished} terminadas en esta tanda, {len(pending)} en cola, corriendo: {names}", flush=True)
                last = time.monotonic()
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nDeteniendo las corridas (al relanzar se repiten desde cero)...", flush=True)
        for proc, _, _ in running.values():
            proc.terminate()
            proc.wait()
    finally:
        keep_awake(False)
        for proc, log, lock in running.values():
            log.close()
            lock.unlink(missing_ok=True)

    if failed:
        print(f"Semillas con error: {failed}")
    return len(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Corre las corridas finales (E12) de varias semillas")
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--level", type=int, required=True, help="nivel de entrenamiento: 2 (§7.1) o 3 (experimento E5)")
    parser.add_argument("--seeds", nargs="+", required=True, help="semillas o rangos, p. ej. 0-3 7")
    parser.add_argument("--workers", type=int, default=1, help="corridas a la vez")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--timesteps", type=int, default=None, help="solo para probar: sobrescribe total_timesteps; usa otro --out")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)  # uso interno: proceso hijo
    args = parser.parse_args()

    out = Path(args.out).resolve()
    seeds = parse_seeds(args.seeds)
    if args.run_one:
        run_one(args.algo, args.level, seeds[0], out, args.timesteps)
        return
    sys.exit(1 if launch(args.algo, args.level, seeds, args.workers, out, args.timesteps) else 0)


if __name__ == "__main__":
    main()
