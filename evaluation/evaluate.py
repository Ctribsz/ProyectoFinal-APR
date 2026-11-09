"""Evaluación final en los niveles 0–3 (especificación §7.2–§7.4; experimentos E3, E5 y E6).

Uso:
    python -m evaluation.evaluate [--runs-dir results/final] [--workers 8] [--episodes 50]

- Evalúa cada corrida terminada de `--runs-dir` (las de `lab.semillas`) y el PID con el mismo código: política
  determinista, `n_episodes` por nivel, semillas fijas y distintas a las de entrenamiento (`configs/evaluation/`).
- Por episodio: retorno, caída, pasos hasta caída, RMSE de v y ψ̇ contra el comando, suavidad, energía de alta
  frecuencia y esfuerzo (§7.4). Las señales para las métricas son las limpias de `info`, como la recompensa.
- Guarda cada corrida en `results/evaluation/<corrida>.json` (se salta si ya existe, salvo con `--force`) y deja
  `docs/finales/evaluacion.md` (IQM con IC 95 % entre semillas, por nivel de entrenamiento y de evaluación) y
  `docs/finales/evaluacion.csv` (una fila por corrida y nivel).
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from baselines.pid_controller import PIDController
from envs.balance_bot_env import REPO_ROOT, BalanceBotEnv
from training.callbacks import action_smoothness, high_freq_energy

EVAL_CONFIG = REPO_ROOT / "configs" / "evaluation" / "evaluation.yaml"
DIAGNOSTICS_CONFIG = REPO_ROOT / "configs" / "training" / "diagnostics.yaml"
DEFAULT_RUNS = REPO_ROOT / "results" / "final"
DEFAULT_OUT = REPO_ROOT / "results" / "evaluation"
DOCS = REPO_ROOT / "docs" / "finales"
METRICS = ("return", "fell", "steps", "rmse_v", "rmse_yaw", "smoothness", "high_freq_energy", "effort")


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: np.ndarray) -> np.ndarray: ...


class SB3Policy:
    """Política de SB3 guardada por `train()`, determinista; aplica la normalización de observaciones si la hubo (D8)."""

    def __init__(self, run_dir: str | Path) -> None:
        from training.train import ALGOS

        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        self.algo: str = cfg["algo"]["algo"]
        self.model = ALGOS[self.algo].load(run_dir / "model.zip", device="cpu")
        self.obs_rms = None
        if (run_dir / "vecnormalize.pkl").exists():
            with (run_dir / "vecnormalize.pkl").open("rb") as f:
                vn = pickle.load(f)
            self.obs_rms, self.clip_obs, self.epsilon = vn.obs_rms, vn.clip_obs, vn.epsilon

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        if self.obs_rms is not None:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
        action, _ = self.model.predict(obs.astype(np.float32), deterministic=True)
        return action


class PIDPolicy:
    """PID en cascada (`baselines/`) con la misma interfaz; se crea por entorno porque depende de su dt y radio."""

    def __init__(self, env: BalanceBotEnv) -> None:
        self.pid = PIDController.from_env(env)

    def reset(self) -> None:
        self.pid.reset()

    def act(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.pid.predict(obs, deterministic=True)
        return action


def load_eval_config(path: str | Path = EVAL_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def cutoff_hz(path: str | Path = DIAGNOSTICS_CONFIG) -> float:
    """El mismo corte de alta frecuencia que los diagnósticos de entrenamiento (§8)."""
    return float(yaml.safe_load(Path(path).read_text())["smoothness"]["cutoff_hz"])


def run_episode(env: BalanceBotEnv, policy: Policy, seed: int, hf_cutoff: float) -> dict[str, float]:
    """Un episodio determinista; devuelve las métricas de §7.4."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    ret, steps, terminated = 0.0, 0, False
    actions, ev, ey = [], [], []
    while True:
        action = np.asarray(policy.act(obs), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += float(reward)
        steps += 1
        actions.append(np.clip(action, -1, 1))
        ev.append(info["v"] - info["command"][0])
        ey.append(info["yaw_rate"] - info["command"][1])
        if terminated or truncated:
            break
    a = np.asarray(actions)
    return {
        "return": ret,
        "fell": float(terminated),
        "steps": float(steps),
        "rmse_v": float(np.sqrt(np.mean(np.square(ev)))),
        "rmse_yaw": float(np.sqrt(np.mean(np.square(ey)))),
        "smoothness": action_smoothness(a),
        "high_freq_energy": high_freq_energy(a, env.dt, hf_cutoff),
        "effort": float(np.mean(np.sum(np.square(a), axis=1))),
    }


def evaluate_levels(make_policy, cfg: dict[str, Any], n_episodes: int | None = None, hf_cutoff: float | None = None) -> dict[str, dict[str, list[float]]]:
    """Métricas por episodio en cada nivel de `cfg["levels"]`. `make_policy(env)` crea la política para ese entorno."""
    n = int(n_episodes or cfg["n_episodes"])
    hf = cutoff_hz() if hf_cutoff is None else hf_cutoff
    out: dict[str, dict[str, list[float]]] = {}
    for level in cfg["levels"]:
        env = BalanceBotEnv(level=int(level))
        policy = make_policy(env)
        episodes = [run_episode(env, policy, int(cfg["seed_offset"]) + k, hf) for k in range(n)]
        out[str(level)] = {m: [e[m] for e in episodes] for m in METRICS}
        env.close()
    return out


def _run_info(run_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return {"run": run_dir.name, "algo": cfg["algo"]["algo"], "train_level": int(cfg["env"]["level"]), "seed": int(cfg["seed"])}


def evaluate_run(run_dir: str | Path, out_dir: str | Path, cfg: dict[str, Any], n_episodes: int | None = None, force: bool = False) -> Path:
    """Evalúa una corrida (o el PID si `run_dir == "pid"`) y guarda `<out_dir>/<nombre>.json`."""
    import torch

    torch.set_num_threads(1)
    if str(run_dir) == "pid":
        info = {"run": "pid", "algo": "pid", "train_level": None, "seed": None}
        make_policy = PIDPolicy
    else:
        run_dir = Path(run_dir)
        info = _run_info(run_dir)
        policy = SB3Policy(run_dir)
        make_policy = lambda env: policy  # noqa: E731  (la red no depende del entorno)
    path = Path(out_dir) / f"{info['run']}.json"
    if path.exists() and not force:
        return path
    info["n_episodes"] = int(n_episodes or cfg["n_episodes"])
    info["levels"] = evaluate_levels(make_policy, cfg, n_episodes)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return path


def finished_runs(runs_dir: Path) -> list[Path]:
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "meta.json").exists() and (p / "model.zip").exists())


# ------------------------------------------------------------------------------------------------ resumen


def summarize(out_dir: Path, docs_dir: Path = DOCS, reps: int = 2000) -> list[Path]:
    """Tabla por nivel de entrenamiento: IQM entre semillas de la media por corrida de cada métrica, con IC 95 %."""
    from analysis.finales import ALGO_ORDER, _ci, iqm_ci

    results = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    pid = next((r for r in results if r["algo"] == "pid"), None)
    runs = [r for r in results if r["algo"] != "pid"]
    docs_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in results:
        for level, m in r["levels"].items():
            rows.append({"run": r["run"], "algo": r["algo"], "train_level": r["train_level"], "seed": r["seed"],
                         "eval_level": int(level), **{k: float(np.mean(v)) for k, v in m.items()}})
    csv_path = docs_dir / "evaluacion.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)

    shown = (("return", "Retorno", 1), ("fell", "Tasa de caídas", 2), ("rmse_v", "RMSE v (m/s)", 3),
             ("rmse_yaw", "RMSE ψ̇ (rad/s)", 3), ("smoothness", "Suavidad ‖Δa‖", 3), ("effort", "Esfuerzo ‖a‖²", 3))
    lines = ["# Evaluación final en los niveles 0–3 (E3, E5, E6)", "",
             "Generado por `python -m evaluation.evaluate`. Política determinista; "
             f"{results[0]['n_episodes'] if results else '?'} episodios por nivel con las mismas semillas para todas las políticas. "
             "Cada corrida aporta la media de sus episodios; entre corridas (semillas) se reporta el IQM con IC 95 % por bootstrap (§7.5). "
             "El PID es determinista y no tiene semillas de entrenamiento: se reporta su media.", ""]
    for train_level in sorted({r["train_level"] for r in runs}):
        group = [r for r in runs if r["train_level"] == train_level]
        algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in group)]
        n_by_algo = {a: sum(r["algo"] == a for r in group) for a in algos}
        lines += [f"## Entrenados en el nivel {train_level}", "",
                  "Semillas: " + ", ".join(f"{a.upper()} {n}" for a, n in n_by_algo.items()), ""]
        for key, title, nd in shown:
            lines += [f"### {title}", "", "| Nivel | " + " | ".join(a.upper() for a in algos) + " | PID |",
                      "|---|" + "---|" * (len(algos) + 1)]
            for level in map(str, sorted({int(lv) for r in group for lv in r["levels"]})):
                values = {a: np.array([np.mean(r["levels"][level][key]) for r in group if r["algo"] == a]) for a in algos}
                ci = iqm_ci(values, reps)
                pid_v = f"{np.mean(pid['levels'][level][key]):.{nd}f}" if pid and level in pid["levels"] else "—"
                lines.append(f"| {level} | " + " | ".join(_ci(ci[a]["iqm"], nd) for a in algos) + f" | {pid_v} |")
            lines.append("")
    md_path = docs_dir / "evaluacion.md"
    md_path.write_text("\n".join(lines))
    return [md_path, csv_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación final en los niveles 0–3 (E3, E5, E6)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--config", default=str(EVAL_CONFIG))
    parser.add_argument("--episodes", type=int, default=None, help="sobrescribe n_episodes (solo para probar)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="reevaluar aunque ya exista el .json")
    parser.add_argument("--no-pid", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    if not args.summary_only:
        targets: list[Any] = ([] if args.no_pid else ["pid"]) + finished_runs(Path(args.runs_dir))
        print(f"Evaluando {len(targets)} políticas con {args.workers} procesos...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(evaluate_run, t, args.out, cfg, args.episodes, args.force) for t in targets]
            for t, fut in zip(targets, futures):
                print(f"  {Path(str(t)).name}: {fut.result()}", flush=True)
    for p in summarize(Path(args.out)):
        print(p)


if __name__ == "__main__":
    main()
