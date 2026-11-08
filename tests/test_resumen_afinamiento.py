"""Pruebas del resumen de estudios de afinamiento (analysis/resumen_afinamiento.py)."""

import csv

import optuna
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.trial import TrialState, create_trial

from analysis.resumen_afinamiento import near_edge, summarize

DISTS = {
    "learning_rate": FloatDistribution(1e-5, 3e-3, log=True),
    "action_noise.sigma": FloatDistribution(0.02, 0.5, log=True),
    "tau": FloatDistribution(0.001, 0.05, log=True),
    "batch_size": CategoricalDistribution([64, 128, 256, 512]),
}


def _params(lr: float, sigma: float) -> dict:
    return {"learning_rate": lr, "action_noise.sigma": sigma, "tau": 0.01, "batch_size": 256}


def test_near_edge_flags_bounds() -> None:
    spec = {"type": "float", "low": 0.001, "high": 0.05, "log": True}
    assert near_edge(0.0492, spec) == "alto"
    assert near_edge(0.0011, spec) == "bajo"
    assert near_edge(0.01, spec) == ""
    assert near_edge(512, {"type": "categorical", "choices": [64, 128, 256, 512]}) == "alto"


def test_summary_counts_states_and_picks_best(tmp_path) -> None:
    db = tmp_path / "ddpg_level2.db"
    study = optuna.create_study(study_name="ddpg_level2", storage=f"sqlite:///{db.as_posix()}", direction="maximize")
    study.set_user_attr("protocol", {"trial_timesteps": 200000, "seeds": [0], "objective": {"last_k_evals": 3}, "env": {"level": 2, "max_episode_steps": 1000}})
    study.add_trial(create_trial(state=TrialState.FAIL, params=_params(1e-4, 0.1), distributions=DISTS))  # interrumpida
    study.set_user_attr("interrumpidas", [0])
    study.add_trial(create_trial(value=500.0, params=_params(1e-4, 0.1), distributions=DISTS, intermediate_values={100000: 400.0, 200000: 500.0}))
    study.add_trial(create_trial(value=800.0, params=_params(9e-4, 0.236), distributions=DISTS, intermediate_values={200000: 800.0}))
    study.add_trial(create_trial(state=TrialState.PRUNED, params=_params(2e-3, 0.05), distributions=DISTS, intermediate_values={60000: 120.0}))

    md = summarize(db, out_dir=tmp_path / "docs", fig_dir=tmp_path / "img").read_text(encoding="utf-8")

    assert "**3 pruebas válidas**: 2 completas, 1 podadas, 0 fallidas" in md
    assert "1 interrumpidas" in md
    assert "### Mejor prueba: #2 (objetivo 800.0)" in md
    assert "| `action_noise.sigma` | 0.02 – 0.5 (log) | 0.1 | 0.236 |" in md  # 3 cifras significativas
    assert (tmp_path / "img" / "afinamiento_ddpg_level2.png").stat().st_size > 0
    with (tmp_path / "docs" / "ddpg_level2.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["estado"] for r in rows] == ["INTERRUMPIDA", "COMPLETE", "COMPLETE", "PRUNED"]
    assert rows[3]["ultimo_valor_intermedio"] == "120.0" and rows[3]["pasos"] == "60000"
