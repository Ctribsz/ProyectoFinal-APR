"""Pruebas de las herramientas para correr en otra computadora (lab/)."""

import sys

from training.train import REPO_ROOT, load_yaml
from training.tune import TUNE_CONFIG

from lab.afinar import study_paths, worker_command
from lab.empaquetar import included_files


def test_study_paths_match_tune_convention() -> None:
    tune_cfg = load_yaml(TUNE_CONFIG)
    name, db = study_paths("ddpg", tune_cfg)
    level = load_yaml(REPO_ROOT / tune_cfg["env_config"])["level"]
    assert name == f"ddpg_level{level}"
    assert db == REPO_ROOT / tune_cfg["storage_dir"] / f"ddpg_level{level}.db"


def test_worker_command_runs_tune_with_its_worker() -> None:
    cmd = worker_command("ppo", 10, 3)
    assert cmd[0] == sys.executable
    assert cmd[1:] == ["-m", "training.tune", "--algo", "ppo", "--n-trials", "10", "--worker", "3"]


def test_package_has_code_but_not_venv_or_results(tmp_path) -> None:
    for rel in ("training/train.py", "configs/a.yaml", ".venv/Lib/x.py", "results/run/model.zip", "results/x.db", "envs/__pycache__/a.pyc"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    files = {p.relative_to(tmp_path).as_posix() for p in included_files(tmp_path)}
    assert files == {"training/train.py", "configs/a.yaml"}
