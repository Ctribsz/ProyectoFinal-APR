"""Ayudas para correr el afinamiento y las corridas finales en Google Colab (ver `lab/colab.ipynb`).

Uso, desde una celda del notebook:
    python -m lab.colab afinar --algo ddpg --workers 2 --drive /content/drive/MyDrive/APR-colab [--sync-min 10]
    python -m lab.colab semillas --algo ddpg --level 2 --seeds 4-6 --workers 1 --drive /content/drive/MyDrive/APR-colab
    python -m lab.colab gpu [--algo ddpg]

- `semillas` (E12) reutiliza `lab.semillas`. A Drive solo se copian las corridas terminadas (con `meta.json`):
  una corrida cortada no deja archivos viejos que se mezclen con la que la repite. Al empezar, trae de Drive
  las terminadas, así `lab.semillas` las salta.

- La máquina de Colab se borra al desconectarse. El estudio corre en el disco local de Colab (SQLite no
  funciona bien sobre Google Drive) y se copia a `<drive>/results/optuna` cada `--sync-min` minutos y al
  terminar. La carpeta de Drive tiene la misma estructura que el repo: se copia tal cual a la laptop.
- Si la sesión se corta, se vuelve a correr la misma celda: trae el estudio de Drive y `lab.afinar` sigue
  desde ahí (las pruebas a medias pasan a FAIL, lanza solo las que faltan y usa números de worker nuevos).
- `afinar` reutiliza `lab.afinar` (y este, `training.tune`): no hay un pipeline aparte.
"""

from __future__ import annotations

import argparse
import json
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
from lab.semillas import DEFAULT_OUT, INCOMPLETE_DIR, is_done

# Lo que `sync` no copia como archivo normal: los .db van con `copy_db`; el .lock es de la máquina que corre.
EXCLUDED_PATTERNS = ("*.db", "*.db-journal", "*.db-wal", "*.db-shm", "*.db.tmp", "*.lock")


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
        excludes = [f"--exclude={p}" for p in EXCLUDED_PATTERNS]
        subprocess.run(["rsync", "-a", *excludes, f"{src}/", f"{dst}/"], check=True)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*EXCLUDED_PATTERNS))
    for db in src.glob("*.db"):
        copy_db(db, dst / db.name)


def afinar(algo: str, workers: int, drive: Path, sync_min: float) -> None:
    tune_cfg = load_yaml(TUNE_CONFIG)
    _, db = study_paths(algo, tune_cfg)
    local = db.parent
    remote = drive / local.relative_to(REPO_ROOT)

    if not db.exists() and (remote / db.name).exists():
        print(f"Trayendo el estudio de Drive ({remote})...", flush=True)
        sync(remote, local)

    cmd = [sys.executable, "-m", "lab.afinar", "--algo", algo, "--workers", str(workers)]
    run_with_sync(cmd, lambda: sync(local, remote), sync_min)
    print(f"Estudio copiado a {remote}")


def copy_completed_runs(src: Path, dst: Path) -> list[str]:
    """Copia de `src/` a `dst/` las corridas terminadas (con meta.json) que `dst/` aún no tiene, y los logs.

    Cada corrida se copia primero a `<nombre>.tmp` y luego se renombra: si la sesión muere a media copia, en
    `dst/` no queda una carpeta a medias con el nombre de la corrida.
    """
    if not src.exists():
        return []
    dst.mkdir(parents=True, exist_ok=True)
    copied = []
    for run in sorted(p for p in src.iterdir() if p.is_dir() and p.name not in (INCOMPLETE_DIR, "logs")):
        if not is_done(run) or is_done(dst / run.name):
            continue
        tmp = dst / f"{run.name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(run, tmp)
        shutil.rmtree(dst / run.name, ignore_errors=True)  # una copia vieja sin meta.json
        tmp.rename(dst / run.name)
        copied.append(run.name)
    if (src / "logs").exists():
        shutil.copytree(src / "logs", dst / "logs", dirs_exist_ok=True)
    return copied


def semillas(algo: str, level: int, seeds: list[str], workers: int, drive: Path, sync_min: float) -> None:
    local = DEFAULT_OUT
    remote = drive / local.relative_to(REPO_ROOT)
    restored = copy_completed_runs(remote, local)
    if restored:
        print(f"Traídas de Drive (ya terminadas): {restored}", flush=True)

    def push() -> None:
        for name in copy_completed_runs(local, remote):
            print(f"[{time.strftime('%H:%M')}] {name} copiada a Drive", flush=True)

    cmd = [sys.executable, "-m", "lab.semillas", "--algo", algo, "--level", str(level), "--seeds", *seeds, "--workers", str(workers)]
    run_with_sync(cmd, push, sync_min)
    print(f"Corridas terminadas en {remote}")


def run_with_sync(cmd: list[str], do_sync, sync_min: float) -> None:
    """Corre `cmd` y llama a `do_sync` cada `sync_min` minutos, al detenerlo y al terminar."""
    print(f"Copia a Drive cada {sync_min:g} min.", flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    last = time.monotonic()
    try:
        while proc.poll() is None:
            time.sleep(5)
            if time.monotonic() - last >= sync_min * 60:
                do_sync()
                print(f"[{time.strftime('%H:%M')}] copia a Drive al día", flush=True)
                last = time.monotonic()
    except KeyboardInterrupt:
        print("\nDeteniendo (se puede continuar después)...", flush=True)
        proc.send_signal(signal.SIGINT)  # lab.afinar y lab.semillas lo atrapan y detienen a sus procesos
        proc.wait()
    finally:
        do_sync()
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
    s = sub.add_parser("semillas", help="corridas finales (E12) de una lista de semillas, copiadas a Drive")
    s.add_argument("--algo", required=True, choices=sorted(ALGOS))
    s.add_argument("--level", type=int, required=True)
    s.add_argument("--seeds", nargs="+", required=True, help="semillas o rangos, p. ej. 4-6")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--drive", required=True)
    s.add_argument("--sync-min", type=float, default=5, help="minutos entre revisiones de corridas terminadas")
    g = sub.add_parser("gpu", help="compara pasos/s con CPU y con GPU")
    g.add_argument("--algo", default="ddpg", choices=sorted(BENCH_STEPS))
    args = parser.parse_args()

    if args.cmd == "afinar":
        afinar(args.algo, args.workers, Path(args.drive), args.sync_min)
    elif args.cmd == "semillas":
        semillas(args.algo, args.level, args.seeds, args.workers, Path(args.drive), args.sync_min)
    else:
        gpu(args.algo)


if __name__ == "__main__":
    main()
