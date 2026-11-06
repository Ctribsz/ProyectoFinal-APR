"""Pruebas de las ayudas para Google Colab (lab/colab.py)."""

import sqlite3
from contextlib import closing

import optuna

from lab.colab import INTERRUPTED_ATTR, mark_interrupted, next_worker, remaining_trials, sync


def test_sync_copies_files_and_db_without_deleting(tmp_path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "ddpg_level2" / "trial000_seed0").mkdir(parents=True)
    (src / "ddpg_level2" / "trial000_seed0" / "meta.json").write_text("{}")
    with closing(sqlite3.connect(src / "ddpg_level2.db")) as con:
        con.execute("create table t (x int)")
        con.execute("insert into t values (7)")
        con.commit()
    (dst / "otro.txt").parent.mkdir(parents=True)
    (dst / "otro.txt").write_text("se queda")

    sync(src, dst)

    assert (dst / "ddpg_level2" / "trial000_seed0" / "meta.json").read_text() == "{}"
    assert (dst / "otro.txt").exists()
    with closing(sqlite3.connect(dst / "ddpg_level2.db")) as con:
        assert con.execute("select x from t").fetchall() == [(7,)]
    assert not list(dst.glob("*.tmp"))


def test_sync_from_missing_source_does_nothing(tmp_path) -> None:
    sync(tmp_path / "no_existe", tmp_path / "dst")
    assert not (tmp_path / "dst").exists()


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
