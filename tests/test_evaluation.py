"""Pruebas de la evaluación final (entregable E11): el PID y un agente de SB3 pasan por el mismo código."""

import json
import math

import numpy as np
import pytest

from envs.balance_bot_env import BalanceBotEnv
from evaluation import evaluate, push_test
from evaluation.evaluate import METRICS, PIDPolicy, SB3Policy, evaluate_levels, load_eval_config, run_episode
from evaluation.push_test import max_tolerated_force, push_episode, recovery_time
from training.train import load_yaml, train
from tests.test_train import TEST_ALGO_CONFIGS


@pytest.fixture(scope="module")
def cfg() -> dict:
    c = load_eval_config()
    c["levels"] = [0, 2]
    c["push_test"]["n_seeds"] = 1
    c["push_test"]["angles"] = [0.0]
    c["push_test"]["max_force"]["tolerance"] = 1.0
    return c


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory) -> "Path":  # noqa: F821
    common = load_yaml(evaluate.REPO_ROOT / "configs" / "training" / "common.yaml")
    common["eval"]["freq"] = 64
    common["eval"]["n_episodes"] = 1
    env_cfg = load_yaml(evaluate.REPO_ROOT / "configs" / "env" / "level_0.yaml")
    return train("ppo", env_cfg, TEST_ALGO_CONFIGS["ppo"], common, seed=0, timesteps=128, out_root=tmp_path_factory.mktemp("runs"))


def test_config_seeds_differ_from_training() -> None:
    c = load_eval_config()
    offsets = {c["seed_offset"], c["push_test"]["seed_offset"], 100000, 200000}  # evaluación y diagnósticos de train()
    assert len(offsets) == 4 and min(offsets) > 9


def test_pid_episode_metrics() -> None:
    env = BalanceBotEnv(level=0)
    m = run_episode(env, PIDPolicy(env), seed=300000, hf_cutoff=5.0)
    assert set(m) == set(METRICS)
    assert m["fell"] == 0 and m["steps"] == env.max_episode_steps and m["return"] > 900
    assert 0 <= m["high_freq_energy"] <= 1 and m["effort"] >= 0


def test_evaluation_is_deterministic(cfg: dict) -> None:
    a = evaluate_levels(PIDPolicy, cfg, n_episodes=2)
    b = evaluate_levels(PIDPolicy, cfg, n_episodes=2)
    assert a == b and set(a) == {"0", "2"}


def test_sb3_and_pid_share_pipeline(cfg: dict, tiny_run, tmp_path) -> None:
    pol = SB3Policy(tiny_run)
    assert pol.algo == "ppo"
    out = tmp_path / "eval"
    paths = [evaluate.evaluate_run(t, out, cfg, n_episodes=2) for t in ("pid", tiny_run)]
    for p in paths:
        r = json.loads(p.read_text())
        assert set(r["levels"]) == {"0", "2"} and len(r["levels"]["0"]["return"]) == 2
    md, csv_path = evaluate.summarize(out, docs_dir=tmp_path / "docs", reps=50)
    assert "PPO" in md.read_text() and "PID" in md.read_text()
    assert len(csv_path.read_text().strip().splitlines()) == 1 + 2 * 2


def test_recovery_time() -> None:
    dt = 0.01
    pitch = np.array([0.2] * 30 + [0.01] * 60)
    assert recovery_time(pitch, dt, 0.035, 0.5) == pytest.approx(31 * dt)
    assert math.isnan(recovery_time(np.array([0.2, 0.01, 0.2] * 40), dt, 0.035, 0.5))


def test_push_episode_falls_only_with_large_force() -> None:
    env = BalanceBotEnv(level=0)
    pol = PIDPolicy(env)
    assert not push_episode(env, pol, 400000, 4.0, 0.0, 2.0, 3.0)[0]
    assert push_episode(env, pol, 400000, 60.0, 0.0, 2.0, 3.0)[0]


def test_pid_max_force_matches_bisection(cfg: dict) -> None:
    """El PID aguantó ~17 N en el avance (búsqueda aparte); la bisección debe dar un valor en ese orden y consistente."""
    env = BalanceBotEnv(level=0)
    pol = PIDPolicy(env)
    pcfg = cfg["push_test"]
    f = max_tolerated_force(env, pol, 400000, 0.0, pcfg)
    assert 8 < f < pcfg["max_force"]["high"]
    args = (pcfg["push_time"], pcfg["post_push_s"])
    assert not push_episode(env, pol, 400000, f, 0.0, *args)[0]
    assert push_episode(env, pol, 400000, f + pcfg["max_force"]["tolerance"], 0.0, *args)[0]


def test_push_test_run_and_summary(cfg: dict, tiny_run, tmp_path) -> None:
    out = tmp_path / "push"
    for t in ("pid", tiny_run):
        r = json.loads(push_test.push_test_run(t, out, cfg).read_text())
        assert len(r["max_force"]) == 1 and set(r["recovery"]) == {str(f) for f in cfg["push_test"]["recovery"]["forces"]}
    md, _ = push_test.summarize(out, docs_dir=tmp_path / "docs", reps=50)
    assert "PID" in md.read_text() and "PPO" in md.read_text()
