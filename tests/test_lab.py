"""Pruebas de las herramientas para correr en otra computadora (lab/)."""

import os
import sys
import time

import optuna

from training.train import REPO_ROOT, load_yaml
from training.tune import TUNE_CONFIG

from lab.afinar import (
    INTERRUPTED_ATTR,
    another_launcher_running,
    mark_interrupted,
    next_worker,
    remaining_trials,
    split_trials,
    study_paths,
    worker_command,
)
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


def test_next_worker_skips_used_numbers(tmp_path) -> None:
    assert next_worker(tmp_path, "ddpg") == 0
    for name in ("ddpg_worker0.log", "ddpg_worker3.log", "ppo_worker9.log", "ddpg_worker1.log.bak"):
        (tmp_path / name).write_text("")
    assert next_worker(tmp_path, "ddpg") == 4
    assert next_worker(tmp_path, "ppo") == 10


def test_remaining_trials_does_not_count_interrupted() -> None:
    states = ["COMPLETE"] * 10 + ["PRUNED"] * 5 + ["FAIL"] * 3
    assert remaining_trials(states, interrupted=2, n_trials=40) == 40 - 16
    assert remaining_trials(states, interrupted=0, n_trials=10) == 0


def test_mark_interrupted_fails_running_trials_once(tmp_path) -> None:
    db = tmp_path / "s.db"
    study = optuna.create_study(study_name="s", storage=f"sqlite:///{db.as_posix()}", direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=1.0))
    study.ask()  # queda RUNNING, como una prueba de una sesión cortada

    states, interrupted = mark_interrupted("s", db)
    assert sorted(states) == ["COMPLETE", "FAIL"]
    assert interrupted == 1
    assert optuna.load_study(study_name="s", storage=f"sqlite:///{db.as_posix()}").user_attrs[INTERRUPTED_ATTR] == [1]

    study.ask()  # otra sesión cortada
    states, interrupted = mark_interrupted("s", db)
    assert sorted(states) == ["COMPLETE", "FAIL", "FAIL"]
    assert interrupted == 2
    assert remaining_trials(states, interrupted, n_trials=5) == 4


def test_mark_interrupted_without_db(tmp_path) -> None:
    assert mark_interrupted("s", tmp_path / "no.db") == ([], 0)


def test_split_trials_never_exceeds_total() -> None:
    assert split_trials(40, 3) == [14, 13, 13]
    assert split_trials(9, 4) == [3, 2, 2, 2]
    assert split_trials(2, 4) == [1, 1]  # no lanza workers sin pruebas


def test_recent_lock_means_another_launcher(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("lab.afinar.shutil.which", lambda _: None)  # sin pgrep: solo cuenta el .lock
    lock = tmp_path / "ddpg.lock"
    assert not another_launcher_running(lock)
    lock.touch()
    assert another_launcher_running(lock)
    old = time.time() - 3600
    os.utime(lock, (old, old))  # un .lock viejo es de un lanzamiento que murió: no bloquea
    assert not another_launcher_running(lock)
