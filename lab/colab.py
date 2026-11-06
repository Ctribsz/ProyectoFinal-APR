"""Ayudas para correr el afinamiento en Google Colab (ver `lab/colab.ipynb`).

Uso, desde una celda del notebook:
    python -m lab.colab afinar --algo ddpg --workers 2 --drive /content/drive/MyDrive/APR-colab [--sync-min 10]
    python -m lab.colab gpu [--algo ddpg]

- La máquina de Colab se borra al desconectarse. El estudio corre en el disco local de Colab (SQLite no
  funciona bien sobre Google Drive) y se copia a `<drive>/results/optuna` cada `--sync-min` minutos y al
  terminar. La carpeta de Drive tiene la misma estructura que el repo: se copia tal cual a la laptop.
- Si la sesión se corta, se vuelve a correr la misma celda: trae el estudio de Drive, marca como FAIL las
  pruebas que quedaron a medias, usa números de worker nuevos y lanza solo las pruebas que faltan.
- `afinar` reutiliza `lab.afinar` (y este, `training.tune`): no hay un pipeline aparte.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import closing
from pathlib import Path

from envs.balance_bot_env import REPO_ROOT
from training.train import ALGOS, COMMON_CONFIG, load_yaml, train
from training.tune import TUNE_CONFIG

from lab.afinar import study_paths
from lab.benchmark import BENCH_STEPS

DB_PATTERNS = ("*.db", "*.db-journal", "*.db-wal", "*.db-shm", "*.db.tmp")
INTERRUPTED_ATTR = "interrumpidas"  # user_attr del estudio: pruebas que quedaron RUNNING por una sesión cortada
FINISHED_STATES = ("COMPLETE", "PRUNED", "FAIL")


def copy_db(src: Path, dst: Path) -> None:
    """Copia un .db de SQLite aunque otro proceso lo esté escribiendo (API de respaldo de SQLite)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with closing(sqlite3.connect(src)) as source, closing(sqlite3.connect(tmp)) as target:
        source.backup(target)
    tmp.replace(dst)


def sync(src: Path, dst: Path) -> None:
    """Copia `src/` en `dst/`: los .db con `copy_db` y el resto con rsync (solo lo que cambió) si existe.

    No borra nada en `dst/`, así que se puede usar en los dos sentidos (Colab → Drive y Drive → Colab).
    """
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    if shutil.which("rsync"):
        excludes = [f"--exclude={p}" for p in DB_PATTERNS]
        subprocess.run(["rsync", "-a", *excludes, f"{src}/", f"{dst}/"], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*DB_PATTERNS))
    for db in src.glob("*.db"):
        copy_db(db, dst / db.name)


def next_worker(log_dir: Path, algo: str) -> int:
    """Primer número de worker sin usar, según los logs de `lab.afinar` (`<algo>_worker<i>.log`)."""
    used = [int(m.group(1)) for p in log_dir.glob(f"{algo}_worker*.log") if (m := re.fullmatch(rf"{algo}_worker(\d+)\.log", p.name))]
    return max(used) + 1 if used else 0


def remaining_trials(states: list[str], interrupted: int, n_trials: int) -> int:
    """Pruebas que faltan para llegar a `n_trials`. Las interrumpidas (FAIL puestas por `mark_interrupted`) no cuentan."""
    finished = sum(states.count(s) for s in FINISHED_STATES) - interrupted
    return max(0, n_trials - finished)


def mark_interrupted(name: str, db: Path) -> tuple[list[str], int]:
    """Pasa a FAIL las pruebas RUNNING de una sesión cortada y devuelve (estados, total de interrumpidas).

    Solo se llama cuando no hay workers corriendo: si no, marcaría como FAIL pruebas vivas.
    """
    if not db.exists():
        return [], 0
    import optuna  # aquí y no arriba: el resto del módulo no lo necesita

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
        print(f"Pruebas que quedaron a medias y se marcan como FAIL: {[t.number for t in running]}")
    return [t.state.name for t in study.get_trials(deepcopy=False)], len(interrupted)


def workers_alive() -> bool:
    """True si ya hay procesos de `training.tune` corriendo en esta máquina (p. ej. una celda anterior)."""
    if not shutil.which("pgrep"):
        return False
    return subprocess.run(["pgrep", "-f", "training.tune"], capture_output=True).returncode == 0


def afinar(algo: str, workers: int, drive: Path, sync_min: float) -> None:
    tune_cfg = load_yaml(TUNE_CONFIG)
    name, db = study_paths(algo, tune_cfg)
    local = db.parent
    remote = drive / local.relative_to(REPO_ROOT)

    if workers_alive():
        sys.exit("Ya hay workers de training.tune corriendo. Detén la celda anterior (o reinicia el entorno) antes de lanzar otra.")
    if not db.exists() and (remote / db.name).exists():
        print(f"Trayendo el estudio de Drive ({remote})...")
        sync(remote, local)

    states, interrupted = mark_interrupted(name, db)
    remaining = remaining_trials(states, interrupted, int(tune_cfg["n_trials"]))
    if remaining == 0:
        print(f"{name}: ya tiene sus {tune_cfg['n_trials']} pruebas. Nada que correr.")
        sync(local, remote)
        return

    workers = min(workers, remaining)
    per_worker = math.ceil(remaining / workers)
    first = next_worker(local / "logs", algo)
    print(f"{name}: faltan {remaining} pruebas. Workers {first}..{first + workers - 1}. Copia a Drive cada {sync_min:g} min.", flush=True)
    cmd = [sys.executable, "-m", "lab.afinar", "--algo", algo, "--workers", str(workers), "--trials-per-worker", str(per_worker), "--first-worker", str(first)]
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    last = time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(5)
            if time.monotonic() - last >= sync_min * 60:
                sync(local, remote)
                print(f"[{time.strftime('%H:%M')}] copiado a Drive", flush=True)
                last = time.monotonic()
    except KeyboardInterrupt:
        print("\nDeteniendo (el estudio se puede continuar después)...")
        proc.send_signal(signal.SIGINT)  # lab.afinar lo atrapa y detiene a sus workers
        proc.wait()
    finally:
        sync(local, remote)
        print(f"Estudio copiado a {remote}")
    if proc.returncode not in (0, None, -signal.SIGINT):
        sys.exit(proc.returncode)


def gpu(algo: str) -> None:
    """Mide los pasos/s de `algo` con device=cpu y device=cuda, con todo lo demás igual a common.yaml."""
    import torch

    if not torch.cuda.is_available():
        print("Este entorno no tiene GPU. En Colab: Entorno de ejecución → Cambiar tipo de entorno → GPU.")
        return
    tune_cfg = load_yaml(TUNE_CONFIG)
    env_cfg = load_yaml(REPO_ROOT / tune_cfg["env_config"])
    algo_cfg = load_yaml(REPO_ROOT / "configs" / "algos" / f"{algo}.yaml")
    print(f"GPU: {torch.cuda.get_device_name(0)}. {algo}, {BENCH_STEPS[algo]} pasos por medición.")
    with tempfile.TemporaryDirectory() as tmp:
        for device in ("cpu", "cuda"):
            common_cfg = {**load_yaml(COMMON_CONFIG), "device": device}
            run_dir = train(algo, env_cfg, algo_cfg, common_cfg, seed=0, timesteps=BENCH_STEPS[algo], out_root=tmp, run_name=f"{algo}_{device}")
            meta = json.loads((run_dir / "meta.json").read_text())
            print(f"   device={device:4s} {meta['steps_per_s']:7.0f} pasos/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Afinamiento en Google Colab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("afinar", help="corre o continúa el estudio y lo copia a Drive")
    p.add_argument("--algo", required=True, choices=sorted(ALGOS))
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--drive", required=True, help="carpeta de Google Drive donde se guarda el estudio")
    p.add_argument("--sync-min", type=float, default=10, help="minutos entre copias a Drive")
    g = sub.add_parser("gpu", help="compara pasos/s con CPU y con GPU")
    g.add_argument("--algo", default="ddpg", choices=sorted(BENCH_STEPS))
    args = parser.parse_args()

    if args.cmd == "afinar":
        afinar(args.algo, args.workers, Path(args.drive), args.sync_min)
    else:
        gpu(args.algo)


if __name__ == "__main__":
    main()
