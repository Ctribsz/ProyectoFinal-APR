"""Pruebas de las ayudas para Google Colab (lab/colab.py)."""

import sqlite3
from contextlib import closing

from lab.colab import sync


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
