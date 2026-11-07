"""Lanza varios procesos de `training.tune` contra el mismo estudio y muestra el avance.

Uso:
    python -m lab.afinar --algo ddpg [--workers 4] [--trials-per-worker 10] [--first-worker 0]

- Cada proceso corre `python -m training.tune --algo <algo> --n-trials <k> --worker <i>`; su salida va a
  `<storage_dir>/logs/<algo>_worker<i>.log`.
- Mientras corre, Windows no suspende la PC (SetThreadExecutionState: dura solo lo que dura este proceso,
  no cambia ninguna configuración). Bloquear la pantalla no detiene nada; cerrar sesión o apagar, sí.
- Si se interrumpe, se vuelve a lanzar con el mismo comando: las pruebas que quedaron a medias pasan a FAIL
  (no cuentan para `n_trials`), se lanzan solo las que faltan y los workers usan números nuevos, para que
  los muestreadores no repitan semillas. `--trials-per-worker` y `--first-worker` desactivan ese cálculo.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
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
HEARTBEAT_S = 120  # un .lock tocado hace menos que esto indica otro `lab.afinar` vivo del mismo algoritmo
INTERRUPTED_ATTR = "interrumpidas"  # user_attr del estudio: pruebas que quedaron RUNNING por una interrupción
FINISHED_STATES = ("COMPLETE", "PRUNED", "FAIL")


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


def next_worker(log_dir: Path, algo: str) -> int:
    """Primer número de worker sin usar, según los logs (`<algo>_worker<i>.log`)."""
    used = [int(m.group(1)) for p in log_dir.glob(f"{algo}_worker*.log") if (m := re.fullmatch(rf"{algo}_worker(\d+)\.log", p.name))]
    return max(used) + 1 if used else 0


def remaining_trials(states: list[str], interrupted: int, n_trials: int) -> int:
    """Pruebas que faltan para llegar a `n_trials`. Las interrumpidas (FAIL puestas por `mark_interrupted`) no cuentan."""
    finished = sum(states.count(s) for s in FINISHED_STATES) - interrupted
    return max(0, n_trials - finished)


def split_trials(total: int, workers: int) -> list[int]:
    """Reparte `total` pruebas entre a lo más `workers` procesos, sin pasarse del total."""
    workers = max(1, min(workers, total))
    return [total // workers + (1 if i < total % workers else 0) for i in range(workers)]


def mark_interrupted(name: str, db: Path) -> tuple[list[str], int]:
    """Pasa a FAIL las pruebas RUNNING de una interrupción y devuelve (estados, total de interrumpidas).

    Solo se llama cuando no hay workers vivos (ver `another_launcher_running`): si no, cortaría pruebas vivas.
    """
    if not db.exists():
        return [], 0
    try:
        study = optuna.load_study(study_name=name, storage=f"sqlite:///{db.as_posix()}")
    except KeyError:  # el .db existe pero el estudio aún no se creó
        return [], 0
    running = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.RUNNING,))
    for trial in running:
        study._storage.set_trial_state_values(trial._trial_id, state=optuna.trial.TrialState.FAIL)
    interrupted = sorted(set(study.user_attrs.get(INTERRUPTED_ATTR, [])) | {t.number for t in running})
    if running:
        study.set_user_attr(INTERRUPTED_ATTR, interrupted)
        print(f"Pruebas que quedaron a medias y pasan a FAIL (no cuentan): {[t.number for t in running]}")
    return [t.state.name for t in study.get_trials(deepcopy=False)], len(interrupted)


def another_launcher_running(lock: Path) -> bool:
    """True si otro `lab.afinar` de este algoritmo sigue vivo (su .lock es reciente) o hay workers sueltos (Linux)."""
    if lock.exists() and time.time() - lock.stat().st_mtime < HEARTBEAT_S:
        return True
    if shutil.which("pgrep"):
        return subprocess.run(["pgrep", "-f", "training.tune"], capture_output=True).returncode == 0
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Afina un algoritmo con varios procesos en paralelo")
    parser.add_argument("--algo", required=True, choices=sorted(ALGOS))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--trials-per-worker", type=int, default=None, help="por defecto, las pruebas que faltan para n_trials de tune.yaml, repartidas")
    parser.add_argument("--first-worker", type=int, default=None, help="por defecto, el primer número sin usar según los logs")
    args = parser.parse_args()

    tune_cfg = load_yaml(TUNE_CONFIG)
    name, db = study_paths(args.algo, tune_cfg)
    log_dir = db.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = log_dir / f"{args.algo}.lock"
    if another_launcher_running(lock):
        sys.exit(f"Ya hay un afinamiento corriendo en esta máquina ({lock.name} reciente o workers de training.tune vivos). "
                 "Espera a que termine o detenlo antes de lanzar otro.")
    lock.touch()

    states, n_interrupted = mark_interrupted(name, db)
    if args.trials_per_worker:
        trials = [args.trials_per_worker] * args.workers
    else:
        remaining = remaining_trials(states, n_interrupted, int(tune_cfg["n_trials"]))
        if remaining == 0:
            lock.unlink(missing_ok=True)
            print(f"{name}: ya tiene sus {tune_cfg['n_trials']} pruebas. Nada que correr. {status_line(name, db)}")
            return
        trials = split_trials(remaining, args.workers)
    first = args.first_worker if args.first_worker is not None else next_worker(log_dir, args.algo)

    env = {**os.environ, "PYTHONUTF8": "1"}
    procs: list[tuple[int, subprocess.Popen]] = []
    logs = []
    print(f"{name}: {sum(trials)} pruebas en {len(trials)} procesos (workers {first}..{first + len(trials) - 1}). Estudio en {db}")
    print(f"Logs en {log_dir}. Ctrl+C detiene todo (el estudio se puede continuar después).", flush=True)
    interrupted = False
    keep_awake(True)
    try:
        for i, n in enumerate(trials):
            worker = first + i
            log = (log_dir / f"{args.algo}_worker{worker}.log").open("a", encoding="utf-8")
            logs.append(log)
            procs.append((worker, subprocess.Popen(worker_command(args.algo, n, worker), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)))
            lock.touch()
            time.sleep(STAGGER_S)

        last = 0.0
        while any(p.poll() is None for _, p in procs):
            lock.touch()  # latido: otro `lab.afinar` del mismo algoritmo no arranca mientras este viva
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
        lock.unlink(missing_ok=True)

    print(f"\nFinal: {status_line(name, db)}")
    failed = [w for w, p in procs if p.returncode not in (0, None)]
    if failed and not interrupted:
        print(f"Procesos con error: {failed}. Revisa sus logs en {log_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
