"""Pruebas del lanzador de corridas finales (lab/semillas.py)."""

import os
import time

from lab.semillas import INCOMPLETE_DIR, child_command, launch, parse_seeds, plan, run_name, set_aside


def test_parse_seeds_accepts_ranges_without_repeats() -> None:
    assert parse_seeds(["0-3", "7"]) == [0, 1, 2, 3, 7]
    assert parse_seeds(["5", "4-6", "5"]) == [5, 4, 6]


def test_plan_separates_done_running_and_pending(tmp_path) -> None:
    done = tmp_path / run_name("ddpg", 2, 0)
    done.mkdir()
    (done / "meta.json").write_text("{}")
    (tmp_path / run_name("ddpg", 2, 1)).mkdir()  # cortada: sin meta.json y sin lock → pendiente
    (tmp_path / f"{run_name('ddpg', 2, 2)}.lock").touch()  # otro lanzador la está corriendo
    stale = tmp_path / f"{run_name('ddpg', 2, 3)}.lock"
    stale.touch()
    os.utime(stale, (time.time() - 3600,) * 2)  # lock viejo: ese lanzador murió

    pending, finished, busy = plan(tmp_path, "ddpg", 2, [0, 1, 2, 3, 4])
    assert finished == [0]
    assert busy == [2]
    assert pending == [1, 3, 4]


def test_set_aside_keeps_the_cut_run(tmp_path) -> None:
    run = tmp_path / run_name("ppo", 3, 5)
    run.mkdir()
    (run / "config.yaml").write_text("x")
    moved = set_aside(run, tmp_path)
    assert not run.exists()
    assert moved.parent == tmp_path / INCOMPLETE_DIR
    assert (moved / "config.yaml").read_text() == "x"


def test_child_command_runs_one_seed() -> None:
    cmd = child_command("td3", 2, 7, "out", None)
    assert cmd[1:3] == ["-m", "lab.semillas"]
    assert cmd[cmd.index("--seeds") + 1] == "7"
    assert "--run-one" in cmd and "--timesteps" not in cmd


def test_launch_runs_pending_and_skips_done(tmp_path) -> None:
    out = tmp_path / "final"
    assert launch("ppo", 0, [0, 1], workers=2, out=out, timesteps=256) == 0
    for seed in (0, 1):
        assert (out / run_name("ppo", 0, seed) / "meta.json").exists()
        assert not (out / f"{run_name('ppo', 0, seed)}.lock").exists()

    (out / run_name("ppo", 0, 2)).mkdir()  # corrida cortada de una sesión anterior
    assert launch("ppo", 0, [0, 1, 2], workers=1, out=out, timesteps=256) == 0
    assert (out / run_name("ppo", 0, 2) / "meta.json").exists()
    assert len(list((out / INCOMPLETE_DIR).iterdir())) == 1
