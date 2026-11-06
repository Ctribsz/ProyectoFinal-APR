"""Lanza varios procesos de `training.tune` contra el mismo estudio y muestra el avance.

Uso:
    python -m lab.afinar --algo ddpg [--workers 4] [--trials-per-worker 10] [--first-worker 0]

- Cada proceso corre `python -m training.tune --algo <algo> --n-trials <k> --worker <i>`; su salida va a
  `<storage_dir>/logs/<algo>_worker<i>.log`.
- Mientras corre, Windows no suspende la PC (SetThreadExecutionState: dura solo lo que dura este proceso,
  no cambia ninguna configuración). Bloquear la pantalla no detiene nada; cerrar sesión o apagar, sí.
- Si se interrumpe, se puede volver a lanzar: el estudio continúa desde el `.db`. Conviene usar
  `--first-worker` con números nuevos para que los muestreadores no repitan semillas.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna

from envs.balance_bot_env import REPO_ROOT
from training.train import ALGOS, load_yaml
from training.tune import TUNE_CONFIG

STATUS_EVERY_S = 60
STAGGER_S = 5  # separación entre arranques, para que no creen el estudio en SQLite al mismo tiempo


def study_paths(algo: str, tune_cfg: dict) -> tuple[str, Path]:
    """Nombre del estudio y ruta del `.db`, con la misma convención que `training.tune.run_study`."""
    level = load_yaml(REPO_ROOT / tune_cfg["env_config"])["level"]
    name = f"{algo}_level{level}"
    return name, REPO_ROOT / tune_cfg["storage_dir"] / f"{name}.db"


def worker_command(algo: str, n_trials: int, worker: int) -> list[str]:
    return [sys.executable, "-m", "training.tune", "--algo", algo, "--n-trials", str(n_trials), "--worker", str(worker)]


def keep_awake(on: bool) -> None:
    if os.name == "nt":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def status_line(name: str, db: Path) -> str:
    if not db.exists():
        return "creando el estudio..."
    try:
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{db.as_posix()}")
    except KeyError:
        return "creando el estudio..."
    states = [t.state.name for t in study.trials]
    line = (
        f"{states.count('COMPLETE')} completas, {states.count('PRUNED')} podadas, "
        f"{states.count('FAIL')} fallidas, {states.count('RUNNING')} corriendo"
    )
    if states.count("COMPLETE"):
        line += f" | mejor: #{study.best_trial.number} con {study.best_value:.1f}"
    return line


def main() -> None:
    parser = argparse.ArgumentParser(description="Afina un algoritmo con varios procesos en paralelo")
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--trials-per-worker", type=int, default=None, help="por defecto n_trials de tune.yaml / workers")
    parser.add_argument("--first-worker", type=int, default=0)
    args = parser.parse_args()

    tune_cfg = load_yaml(TUNE_CONFIG)
    per_worker = args.trials_per_worker or -(-int(tune_cfg["n_trials"]) // args.workers)  # división hacia arriba
    name, db = study_paths(args.algo, tune_cfg)
    log_dir = db.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONUTF8": "1"}
    procs: list[tuple[int, subprocess.Popen]] = []
    logs = []
    print(f"{name}: {args.workers} procesos × {per_worker} pruebas. Estudio en {db}")
    print(f"Logs en {log_dir}. Ctrl+C detiene todo (el estudio se puede continuar después).")
    interrupted = False
    keep_awake(True)
    try:
        for i in range(args.workers):
            worker = args.first_worker + i
            log = (log_dir / f"{args.algo}_worker{worker}.log").open("a", encoding="utf-8")
            logs.append(log)
            procs.append((worker, subprocess.Popen(worker_command(args.algo, per_worker, worker), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)))
            time.sleep(STAGGER_S)

        last = 0.0
        while any(p.poll() is None for _, p in procs):
            if time.monotonic() - last >= STATUS_EVERY_S:
                running = sum(p.poll() is None for _, p in procs)
                print(f"[{datetime.now():%H:%M}] {status_line(name, db)} | procesos vivos: {running}/{len(procs)}", flush=True)
                last = time.monotonic()
            time.sleep(2)
    except KeyboardInterrupt:
        interrupted = True
        print("\nDeteniendo los procesos...")
        for _, p in procs:
            p.terminate()
            p.wait()
    finally:
        keep_awake(False)
        for log in logs:
            log.close()

    print(f"\nFinal: {status_line(name, db)}")
    failed = [w for w, p in procs if p.returncode not in (0, None)]
    if failed and not interrupted:
        print(f"Procesos con error: {failed}. Revisa sus logs en {log_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
