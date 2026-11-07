"""Pruebas de las ayudas para Google Colab (lab/colab.py)."""

import sqlite3
from contextlib import closing

from lab.colab import copy_completed_runs, sync


def test_copy_completed_runs_only_copies_finished_ones(tmp_path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name, done in (("ddpg_level2_seed0", True), ("ddpg_level2_seed1", False)):
        (src / name).mkdir(parents=True)
        (src / name / "config.yaml").write_text("x")
        if done:
            (src / name / "meta.json").write_text("{}")
    (src / "_incompletas" / "viejo").mkdir(parents=True)
    (src / "logs").mkdir()
    (src / "logs" / "ddpg_level2_seed1.log").write_text("log")
    (dst / "ddpg_level2_seed0").mkdir(parents=True)  # copia a medias de una sesión anterior

    assert copy_completed_runs(src, dst) == ["ddpg_level2_seed0"]
    assert (dst / "ddpg_level2_seed0" / "meta.json").exists()
    assert not (dst / "ddpg_level2_seed1").exists()
    assert not (dst / "_incompletas").exists()
    assert (dst / "logs" / "ddpg_level2_seed1.log").read_text() == "log"
    assert not list(dst.glob("*.tmp"))
    assert copy_completed_runs(src, dst) == []  # la segunda vez no copia nada


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
