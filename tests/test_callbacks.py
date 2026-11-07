"""Pruebas de los diagnósticos de §8 (training/callbacks.py)."""

import numpy as np
import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from training.callbacks import action_smoothness, discounted_returns, high_freq_energy
from training.train import COMMON_CONFIG, DIAGNOSTICS_CONFIG, REPO_ROOT, load_yaml, train


def test_action_smoothness_is_mean_step_norm() -> None:
    actions = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    assert action_smoothness(actions) == pytest.approx(2.5)  # (5 + 0) / 2
    assert action_smoothness(actions[:1]) == 0.0


def test_high_freq_energy_separates_slow_and_fast_signals() -> None:
    dt = 0.01
    t = np.arange(1000) * dt
    slow = np.sin(2 * np.pi * 1.0 * t)[:, None]
    fast = np.sin(2 * np.pi * 20.0 * t)[:, None]
    assert high_freq_energy(slow, dt, cutoff_hz=5.0) < 0.01
    assert high_freq_energy(fast, dt, cutoff_hz=5.0) > 0.99
    assert high_freq_energy(np.ones((100, 2)), dt, cutoff_hz=5.0) == 0.0  # constante: sin energía fuera de 0 Hz


def test_discounted_returns() -> None:
    assert discounted_returns(np.array([1.0, 1.0, 1.0]), 0.5) == pytest.approx([1.75, 1.5, 1.0])


def _run(algo: str, tmp_path, timesteps: int = 1300, freq: int = 600):
    diag = load_yaml(DIAGNOSTICS_CONFIG) | {"freq": freq, "n_episodes": 1}
    env_cfg = load_yaml(REPO_ROOT / "configs" / "env" / "level_0.yaml")
    run = train(algo, env_cfg, load_yaml(REPO_ROOT / "configs" / "algos" / f"{algo}.yaml"), load_yaml(COMMON_CONFIG),
                seed=0, timesteps=timesteps, out_root=tmp_path, run_name=algo, diagnostics_cfg=diag)
    with np.load(run / "diagnostics.npz") as data:
        diagnostics = {k: data[k] for k in data.files}
    tags = EventAccumulator(str(next((run / "tb").iterdir()))).Reload().Tags()["scalars"]
    return run, diagnostics, set(tags)


@pytest.mark.parametrize("algo", ["ddpg", "td3", "sac"])
def test_off_policy_runs_log_q_overestimation_and_smoothness(algo, tmp_path) -> None:
    run, diag, tags = _run(algo, tmp_path)
    assert list(diag["timesteps"]) == [600, 1200]
    assert np.all(np.isfinite(diag["q_overestimation"]))
    assert np.allclose(diag["q_overestimation"], diag["q_estimate"] - diag["q_return"])
    assert {"diagnostics/q_overestimation", "diagnostics/action_smoothness", "diagnostics/high_freq_energy"} <= tags
    assert "diagnostics" in load_yaml(run / "config.yaml")  # la corrida guarda su configuración de diagnósticos
    if algo == "sac":
        assert "train/ent_coef" in tags  # α lo registra SB3


def test_ppo_logs_smoothness_and_native_kl_clip(tmp_path) -> None:
    _, diag, tags = _run("ppo", tmp_path, timesteps=4200, freq=2000)  # SB3 registra KL/clip tras cada rollout de 2048
    assert np.all(np.isnan(diag["q_overestimation"]))  # PPO no tiene crítico Q
    assert np.all(np.isfinite(diag["action_smoothness"]))
    assert {"diagnostics/action_smoothness", "train/approx_kl", "train/clip_fraction"} <= tags
    assert "diagnostics/q_overestimation" not in tags


def test_training_without_diagnostics_has_no_extra_files(tmp_path) -> None:
    env_cfg = load_yaml(REPO_ROOT / "configs" / "env" / "level_0.yaml")
    run = train("ppo", env_cfg, load_yaml(REPO_ROOT / "configs" / "algos" / "ppo.yaml"), load_yaml(COMMON_CONFIG),
                seed=0, timesteps=256, out_root=tmp_path, run_name="sin")
    assert not (run / "diagnostics.npz").exists()
    assert "diagnostics" not in load_yaml(run / "config.yaml")
