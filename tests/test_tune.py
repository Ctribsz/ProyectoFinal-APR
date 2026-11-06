"""Pruebas del afinamiento con Optuna (entregable E9). Estudios de 2 pruebas muy cortas."""

import json

import numpy as np
import optuna
import pytest
import yaml

from envs import load_config
from training.train import REPO_ROOT, build_hyperparams, load_yaml, train
from training.tune import TUNE_CONFIG, TrialPruningCallback, apply_params, objective_value, run_study, sample_params

# Espacio pequeño con una categórica mixta, el caso delicado de configs/algos/ (como ent_coef de SAC).
TEST_SAC_CONFIG = {
    "algo": "sac",
    "hyperparams": {"learning_starts": 50},
    "search_space": {
        "learning_rate": {"type": "float", "low": 1.0e-4, "high": 1.0e-3, "log": True},
        "ent_coef": {"type": "categorical", "choices": ["auto", 0.01, 0.1]},
    },
}


@pytest.fixture
def common() -> dict:
    cfg = load_yaml(REPO_ROOT / "configs" / "training" / "common.yaml")
    cfg["net_arch"] = [16, 16]
    cfg["eval"] = {"freq": 100, "n_episodes": 2, "seed_offset": 1000}
    return cfg


@pytest.fixture
def tune_cfg() -> dict:
    cfg = load_yaml(TUNE_CONFIG)
    cfg["sampler"]["n_startup_trials"] = 1
    cfg["pruner"] = {"type": "median", "n_startup_trials": 1, "n_warmup_steps": 0}
    cfg["objective"]["last_k_evals"] = 2
    return cfg


def test_repo_search_spaces_sample_and_build() -> None:
    """Los search_space de los 4 YAML se muestrean y producen hiperparámetros válidos para SB3."""
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for path in sorted((REPO_ROOT / "configs" / "algos").glob("*.yaml")):
        algo_cfg = load_yaml(path)
        for _ in range(5):
            params = sample_params(study.ask(), algo_cfg["search_space"])
            assert set(params) == set(algo_cfg["search_space"])
            build_hyperparams(apply_params(algo_cfg, params), 2)


def test_dotted_keys_are_nested() -> None:
    cfg = {"algo": "ddpg", "hyperparams": {"tau": 0.005, "action_noise": {"type": "normal", "sigma": 0.1}}}
    out = apply_params(cfg, {"action_noise.sigma": 0.3, "tau": 0.02})
    assert out["hyperparams"]["action_noise"] == {"type": "normal", "sigma": 0.3}
    assert out["hyperparams"]["tau"] == 0.02
    assert "action_noise.sigma" not in out["hyperparams"]
    assert cfg["hyperparams"]["action_noise"]["sigma"] == 0.1  # el original no se modifica


def test_mixed_categorical_keeps_types() -> None:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = {sample_params(study.ask(), TEST_SAC_CONFIG["search_space"])["ent_coef"] for _ in range(30)}
    assert seen == {"auto", 0.01, 0.1}


def test_sample_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        sample_params(optuna.create_study().ask(), {"x": {"type": "bool"}})


def test_objective_is_mean_of_last_k() -> None:
    results = [[0.0, 2.0], [10.0, 10.0], [20.0, 30.0]]
    assert objective_value(results, 2) == pytest.approx(17.5)
    assert objective_value(results, 10) == pytest.approx(12.0)


def test_study_writes_db_and_runs(common: dict, tune_cfg: dict, tmp_path) -> None:
    study = run_study("sac", load_config(2), TEST_SAC_CONFIG, common, tune_cfg, n_trials=2, trial_timesteps=200, storage_dir=tmp_path)

    db = tmp_path / "sac_level2.db"
    assert db.exists()
    reloaded = optuna.load_study(study_name="sac_level2", storage=f"sqlite:///{db.as_posix()}")
    assert len(reloaded.trials) == 2
    assert reloaded.user_attrs["protocol"]["trial_timesteps"] == 200

    for trial in reloaded.trials:
        assert trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        (run_dir,) = trial.user_attrs["run_dirs"]
        config = yaml.safe_load((tmp_path / "sac_level2" / f"trial{trial.number:03d}_seed0" / "config.yaml").read_text())
        assert str(tmp_path / "sac_level2") in run_dir
        assert config["algo"]["hyperparams"]["learning_rate"] == trial.params["learning_rate"]
        assert config["algo"]["hyperparams"]["ent_coef"] == trial.params["ent_coef"]
        assert config["env"]["level"] == 2
        meta = json.loads((tmp_path / "sac_level2" / f"trial{trial.number:03d}_seed0" / "meta.json").read_text())
        assert meta["seed"] == 0 and "versions" in meta


def test_study_resumes_and_rejects_other_protocol(common: dict, tune_cfg: dict, tmp_path) -> None:
    args = ("sac", load_config(2), TEST_SAC_CONFIG, common, tune_cfg)
    run_study(*args, n_trials=1, trial_timesteps=200, storage_dir=tmp_path)
    study = run_study(*args, n_trials=1, trial_timesteps=200, storage_dir=tmp_path, worker=1)
    assert len(study.trials) == 2
    with pytest.raises(ValueError):
        run_study(*args, n_trials=1, trial_timesteps=300, storage_dir=tmp_path)


def test_pruning_stops_run_and_keeps_it(common: dict, tmp_path) -> None:
    # Umbral inalcanzable: la primera evaluación ya poda.
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.ThresholdPruner(lower=1e9))
    trial = study.ask()
    pruning = TrialPruningCallback(trial, last_k=3)
    run_dir = train("sac", load_config(0), TEST_SAC_CONFIG, common, seed=0, timesteps=1000, out_root=tmp_path, callback_after_eval=pruning)

    assert pruning.pruned
    assert len(np.load(run_dir / "evaluations.npz")["results"]) == 1
    assert (run_dir / "model.zip").exists()
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["total_timesteps"] == 1000 and meta["timesteps_done"] < 1000
    expected = objective_value(np.load(run_dir / "evaluations.npz")["results"], 3)
    assert study.trials[0].intermediate_values == {100: pytest.approx(expected)}
