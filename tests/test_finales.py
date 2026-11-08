"""Pruebas del análisis de corridas finales (analysis/finales.py), con corridas sintéticas."""

import csv
import json

import numpy as np
import pytest
import yaml
from torch.utils.tensorboard import SummaryWriter

from analysis.finales import Run, analyze, final_return, iqm_ci, load_runs, steps_to_fraction


def _fake_run(root, algo: str, seed: int, level: int = 2, plateau: float = 800.0, with_q: bool = True) -> None:
    run = root / f"{algo}_level{level}_seed{seed}"
    (run / "tb" / "run_1").mkdir(parents=True)
    steps = np.arange(10000, 110000, 10000)
    returns = plateau * (1 - np.exp(-steps / 30000)) + seed
    (run / "config.yaml").write_text(yaml.safe_dump({"algo": {"algo": algo}, "env": {"level": level, "max_episode_steps": 1000}, "seed": seed}))
    (run / "meta.json").write_text(json.dumps({"duration_s": 3600.0, "steps_per_s": 30.0, "machine": {"hostname": "pc"}, "git": {"commit": "abc1234"}}))
    np.savez(run / "evaluations.npz", timesteps=steps, results=np.repeat(returns[:, None], 10, axis=1))
    d_steps = np.arange(50000, 110000, 50000)
    q = np.full(len(d_steps), 10.0 + seed) if with_q else np.full(len(d_steps), np.nan)
    np.savez(run / "diagnostics.npz", timesteps=d_steps, q_overestimation=q, action_smoothness=np.full(len(d_steps), 0.1),
             high_freq_energy=np.full(len(d_steps), 0.2), q_estimate=q, q_return=q)
    writer = SummaryWriter(str(run / "tb" / "run_1"))
    for i, (s, r) in enumerate(zip(steps, returns)):
        writer.add_scalar("eval/mean_reward", r, int(s), walltime=1000.0 + 360.0 * (i + 1))
    writer.close()


def test_final_return_and_sample_efficiency() -> None:
    run = Run(path=None, algo="ppo", level=2, seed=0, hostname="pc", duration_s=1, steps_per_s=1, commit=None,
              timesteps=np.array([10, 20, 30, 40]), returns=np.array([100.0, 500.0, 900.0, 1000.0]))
    assert final_return(run, last_k=2) == 950.0
    assert steps_to_fraction(run, last_k=2) == 30  # 90 % de 950 = 855: primero en el paso 30
    run.returns = np.array([-5.0, -3.0, -1.0, -2.0])
    assert np.isnan(steps_to_fraction(run, last_k=2))


def test_iqm_ci_brackets_the_point() -> None:
    stats = iqm_ci({"a": np.array([1.0, 2.0, 3.0, 4.0, 100.0])}, reps=200)["a"]
    point, low, high = stats["iqm"]
    assert low <= point <= high
    assert point == pytest.approx(3.0)  # el IQM descarta el atípico 100: media de 2, 3 y 4


def test_load_runs_reads_config_and_wall_clock(tmp_path) -> None:
    _fake_run(tmp_path, "ddpg", 0)
    (tmp_path / "ddpg_level2_seed1").mkdir()  # cortada: sin meta.json, no cuenta
    runs = load_runs(tmp_path)
    assert [(r.algo, r.level, r.seed) for r in runs] == [("ddpg", 2, 0)]
    # último evento a 1000 + 3600 s, duración 3600 s → inicio en 1000 s; evaluaciones cada 0.1 h
    assert runs[0].wall_h == pytest.approx(np.arange(1, 11) * 0.1)


def test_analyze_writes_report_per_level(tmp_path) -> None:
    runs_dir = tmp_path / "final"
    for seed in range(5):
        _fake_run(runs_dir, "ddpg", seed)
        _fake_run(runs_dir, "ppo", seed, plateau=950.0, with_q=False)
    _fake_run(runs_dir, "ppo", 0, level=3)
    written = analyze(runs_dir, reps=100, out_dir=tmp_path / "docs", fig_dir=tmp_path / "img")

    assert [p.name for p in written] == ["level2.md", "level3.md"]
    md = (tmp_path / "docs" / "level2.md").read_text(encoding="utf-8")
    assert "| DDPG | 5: [0, 1, 2, 3, 4]" in md and "| PPO | 5: [0, 1, 2, 3, 4]" in md
    assert "semillas repetidas" not in md
    assert "Sobreestimación de Q" in md and "finales_level2_q.png" in md
    assert "menos de 5 semillas" in (tmp_path / "docs" / "level3.md").read_text(encoding="utf-8")
    for name in ("finales_level2_pasos.png", "finales_level2_reloj.png", "finales_level2_q.png"):
        assert (tmp_path / "img" / name).stat().st_size > 0
    with (tmp_path / "docs" / "level2.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10 and {r["algo"] for r in rows} == {"ddpg", "ppo"}


def test_repeated_seed_is_flagged(tmp_path) -> None:
    for name in ("a", "b"):  # la misma semilla corrida en dos máquinas
        _fake_run(tmp_path / name, "ddpg", 3)
        (tmp_path / name / "ddpg_level2_seed3").rename(tmp_path / f"ddpg_level2_seed3_{name}")
    analyze(tmp_path, reps=50, out_dir=tmp_path / "docs", fig_dir=tmp_path / "img")
    assert "semillas repetidas (DDPG 3)" in (tmp_path / "docs" / "level2.md").read_text(encoding="utf-8")
