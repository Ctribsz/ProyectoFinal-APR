"""Pruebas del pipeline común de entrenamiento (entregable E6). Corridas muy cortas de los 4 algoritmos."""

import copy
import json

import numpy as np
import pytest
import yaml
from stable_baselines3.common.noise import NormalActionNoise

from envs import load_config
from training.train import ALGOS, REPO_ROOT, build_hyperparams, load_yaml, train

# DDPG y PPO son de Mario (E5); aquí se usan configuraciones mínimas solo para probar el pipeline.
TEST_ALGO_CONFIGS = {
    "ddpg": {"algo": "ddpg", "hyperparams": {"learning_starts": 50, "action_noise": {"type": "normal", "sigma": 0.1}}},
    "td3": {"algo": "td3", "hyperparams": {"learning_starts": 50, "action_noise": {"type": "normal", "sigma": 0.1}}},
    "sac": {"algo": "sac", "hyperparams": {"learning_starts": 50}},
    "ppo": {"algo": "ppo", "hyperparams": {"n_steps": 64, "batch_size": 32, "n_epochs": 1}, "n_envs": 2},
}


@pytest.fixture
def common() -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "training" / "common.yaml")
    cfg["net_arch"] = [16, 16]
    cfg["eval"] = {"freq": 100, "n_episodes": 2, "seed_offset": 1000}
    return cfg


@pytest.mark.parametrize("algo", sorted(ALGOS))
def test_train_writes_run(algo: str, common: dict, tmp_path) -> None:
    run_dir = train(algo, load_config(0), TEST_ALGO_CONFIGS[algo], common, seed=3, timesteps=256, out_root=tmp_path)

    for name in ("config.yaml", "meta.json", "evaluations.npz", "model.zip"):
        assert (run_dir / name).exists(), name
    assert any((run_dir / "tb").rglob("events.out.tfevents.*"))

    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    assert config["seed"] == 3 and config["algo"]["algo"] == algo
    assert config["env"] == load_config(0) and config["common"] == common

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["seed"] == 3 and meta["eval_seed"] == 1003
    for lib in ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy"):
        assert meta["versions"][lib] != "desconocida"
    assert "commit" in meta["git"]


def test_model_reloads_and_acts(common: dict, tmp_path) -> None:
    run_dir = train("sac", load_config(0), TEST_ALGO_CONFIGS["sac"], common, seed=0, timesteps=200, out_root=tmp_path)
    model = ALGOS["sac"].load(run_dir / "model.zip", device="cpu")
    action, _ = model.predict(np.zeros(9, dtype=np.float32), deterministic=True)
    assert action.shape == (2,) and np.all(np.abs(action) <= 1)


def test_normalize_saves_stats(common: dict, tmp_path) -> None:
    common["normalize"] = True
    run_dir = train("td3", load_config(0), TEST_ALGO_CONFIGS["td3"], common, seed=0, timesteps=200, out_root=tmp_path)
    assert (run_dir / "vecnormalize.pkl").exists()


def test_same_seed_same_result(common: dict, tmp_path) -> None:
    runs = [
        train("ppo", load_config(0), TEST_ALGO_CONFIGS["ppo"], common, seed=5, timesteps=256, out_root=tmp_path / str(i))
        for i in range(2)
    ]
    r0, r1 = (np.load(r / "evaluations.npz")["results"] for r in runs)
    np.testing.assert_array_equal(r0, r1)


def test_action_noise_conversion() -> None:
    params = build_hyperparams({"hyperparams": {"action_noise": {"type": "normal", "sigma": 0.2}, "tau": 0.01}}, 2)
    assert isinstance(params["action_noise"], NormalActionNoise)
    assert params["tau"] == 0.01


def test_rejects_mismatched_config(common: dict, tmp_path) -> None:
    with pytest.raises(ValueError):
        train("td3", load_config(0), TEST_ALGO_CONFIGS["sac"], common, seed=0, timesteps=10, out_root=tmp_path)


def test_repo_algo_configs_are_valid() -> None:
    """Los YAML de configs/algos/ que existan tienen el formato de docs/fase-2-prueba-de-humo.md (T1)."""
    for path in sorted((REPO_ROOT / "configs" / "algos").glob("*.yaml")):
        cfg = load_yaml(path)
        assert cfg["algo"] == path.stem and cfg["algo"] in ALGOS
        assert isinstance(cfg["hyperparams"], dict)
        assert "net_arch" not in cfg["hyperparams"] and "policy_kwargs" not in cfg["hyperparams"]
        for name, spec in cfg.get("search_space", {}).items():
            assert spec["type"] in ("float", "int", "categorical"), name
        build_hyperparams(cfg, 2)
