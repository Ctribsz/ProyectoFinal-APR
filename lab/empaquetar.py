"""Empaqueta el repositorio en un .zip para llevarlo a otra computadora (USB, correo, OneDrive).

Uso:
    python -m lab.empaquetar [--out ../ProyectoFinal-APR-lab.zip]

Deja fuera lo que no se versiona ni hace falta para correr: .venv, results/, .git, cachés, modelos y bases
de datos. El .zip trae una sola carpeta, ProyectoFinal-APR/, con el código, configs/, assets/, docs/ y lab/.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path

from envs.balance_bot_env import REPO_ROOT

EXCLUDED_DIRS = {".venv", "venv", ".git", "results", "runs", "logs", "wandb", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".vscode", ".idea", ".claude"}
EXCLUDED_SUFFIXES = {".zip", ".db", ".mp4", ".gif", ".pyc", ".pkl"}


def included_files(root: Path) -> list[Path]:
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_DIRS)  # no entra a .venv ni a results/
        files += [Path(dirpath) / f for f in sorted(filenames) if Path(f).suffix not in EXCLUDED_SUFFIXES]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Empaqueta el repositorio para otra computadora")
    parser.add_argument("--out", default=str(REPO_ROOT.parent / "ProyectoFinal-APR-lab.zip"))
    args = parser.parse_args()

    out = Path(args.out).resolve()
    files = included_files(REPO_ROOT)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, Path("ProyectoFinal-APR") / path.relative_to(REPO_ROOT))
    print(f"{out}: {len(files)} archivos, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
