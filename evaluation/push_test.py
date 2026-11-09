"""Prueba de empujón (especificación §7.3 E4 y §7.4; hipótesis H2).

Uso:
    python -m evaluation.push_test [--runs-dir results/final] [--workers 8]

En el nivel 0 (sin empujones aleatorios ni comandos) se aplica un único empujón a t = 2 s en el punto alto del
chasis, hacia adelante y hacia atrás, desde `n_seeds` estados iniciales distintos:

- **Fuerza máxima tolerada:** bisección sobre la fuerza entre `low` (no cae) y `high`; si con `high` no cae, se
  reporta `high` como cota. Se supone que caer es monótono en la fuerza.
- **Tiempo de recuperación:** con fuerzas fijas, tiempo desde el inicio del empujón hasta que |θ| < umbral de
  forma sostenida durante `hold_s`. NaN si cae o no se recupera antes del final de la simulación.

Guarda `results/evaluation/push/<corrida>.json` y el resumen `docs/finales/empujon.md` / `.csv`.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from envs.balance_bot_env import REPO_ROOT, BalanceBotEnv
from evaluation.evaluate import DEFAULT_OUT, DEFAULT_RUNS, DOCS, EVAL_CONFIG, PIDPolicy, Policy, SB3Policy, _run_info, finished_runs, load_eval_config


def push_episode(env: BalanceBotEnv, policy: Policy, seed: int, force: float, angle: float, push_time: float, post_push_s: float) -> tuple[bool, np.ndarray]:
    """Un episodio con un solo empujón; devuelve (cayó, |θ| limpio desde el inicio del empujón)."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    env.perturbations.schedule(push_time, force, angle)
    n_steps = int(round((push_time + post_push_s) / env.dt))
    pitch_after = []
    for _ in range(n_steps):
        obs, _, terminated, truncated, info = env.step(np.asarray(policy.act(obs), dtype=np.float32))
        if env.data.time >= push_time - 1e-9:
            pitch_after.append(abs(info["pitch"]))
        if terminated:
            return True, np.asarray(pitch_after)
        if truncated:
            break
    return False, np.asarray(pitch_after)


def recovery_time(abs_pitch: np.ndarray, dt: float, threshold: float, hold_s: float) -> float:
    """Tiempo hasta el primer paso desde el que |θ| < threshold durante `hold_s` seguidos; NaN si no ocurre."""
    hold = int(round(hold_s / dt))
    run = 0
    for i, below in enumerate(abs_pitch < threshold):
        run = run + 1 if below else 0
        if run >= hold:
            return (i - hold + 2) * dt  # el paso i-hold+1 es el primero de la racha; abs_pitch[0] es el paso tras el inicio
    return float("nan")


def max_tolerated_force(env: BalanceBotEnv, policy: Policy, seed: int, angle: float, pcfg: dict[str, Any]) -> float:
    low, high, tol = (float(pcfg["max_force"][k]) for k in ("low", "high", "tolerance"))
    args = (float(pcfg["push_time"]), float(pcfg["post_push_s"]))
    if not push_episode(env, policy, seed, high, angle, *args)[0]:
        return high
    while high - low > tol:
        mid = 0.5 * (low + high)
        if push_episode(env, policy, seed, mid, angle, *args)[0]:
            high = mid
        else:
            low = mid
    return low


def push_test(make_policy, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fuerza máxima y tiempos de recuperación por semilla y dirección."""
    pcfg = cfg["push_test"]
    env = BalanceBotEnv(level=int(pcfg["level"]))
    policy = make_policy(env)
    seeds = [int(pcfg["seed_offset"]) + k for k in range(int(pcfg["n_seeds"]))]
    rcfg = pcfg["recovery"]
    out: dict[str, Any] = {"max_force": [], "recovery": {str(f): [] for f in rcfg["forces"]}, "fell": {str(f): [] for f in rcfg["forces"]}}
    for angle in pcfg["angles"]:
        for seed in seeds:
            out["max_force"].append(max_tolerated_force(env, policy, seed, float(angle), pcfg))
            for f in rcfg["forces"]:
                fell, pitch = push_episode(env, policy, seed, float(f), float(angle), float(pcfg["push_time"]), float(pcfg["post_push_s"]))
                out["fell"][str(f)].append(float(fell))
                out["recovery"][str(f)].append(float("nan") if fell else recovery_time(pitch, env.dt, float(rcfg["threshold"]), float(rcfg["hold_s"])))
    env.close()
    return out


def push_test_run(run_dir: str | Path, out_dir: str | Path, cfg: dict[str, Any], force: bool = False) -> Path:
    import torch

    torch.set_num_threads(1)
    if str(run_dir) == "pid":
        info, make_policy = {"run": "pid", "algo": "pid", "train_level": None, "seed": None}, PIDPolicy
    else:
        info = _run_info(Path(run_dir))
        policy = SB3Policy(run_dir)
        make_policy = lambda env: policy  # noqa: E731
    path = Path(out_dir) / f"{info['run']}.json"
    if path.exists() and not force:
        return path
    info.update(push_test(make_policy, cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info))
    return path


def summarize(out_dir: Path, docs_dir: Path = DOCS, reps: int = 2000) -> list[Path]:
    from analysis.finales import ALGO_ORDER, _ci, iqm_ci

    results = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("*.json"))]
    docs_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        row = {"run": r["run"], "algo": r["algo"], "train_level": r["train_level"], "seed": r["seed"],
               "max_force": float(np.mean(r["max_force"]))}
        for f, v in r["recovery"].items():
            row[f"recovery_{f}N"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            row[f"fell_{f}N"] = float(np.mean(r["fell"][f]))
        rows.append(row)
    csv_path = docs_dir / "empujon.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["run"])
        writer.writeheader()
        writer.writerows(rows)

    pid = next((row for row in rows if row["algo"] == "pid"), None)
    forces = list(results[0]["recovery"]) if results else []
    lines = ["# Prueba de empujón (E4, H2)", "",
             "Generado por `python -m evaluation.push_test`. Nivel 0, empujón único a t = 2 s en el punto alto del chasis, "
             "adelante y atrás, desde varios estados iniciales. Fuerza máxima: bisección (media por corrida). Recuperación: "
             "tiempo hasta |θ| < 2° sostenido 0.5 s (media de los episodios sin caída). Entre semillas: IQM con IC 95 %.", ""]
    for train_level in sorted({r["train_level"] for r in rows if r["algo"] != "pid"}):
        group = [r for r in rows if r["train_level"] == train_level and r["algo"] != "pid"]
        algos = [a for a in ALGO_ORDER if any(r["algo"] == a for r in group)]
        cols = [("max_force", "Fuerza máx. (N)", 1)] + [(f"recovery_{f}N", f"Recuperación a {f} N (s)", 2) for f in forces] \
            + [(f"fell_{f}N", f"Caídas a {f} N", 2) for f in forces]
        lines += [f"## Entrenados en el nivel {train_level}", "", "| Política | " + " | ".join(c[1] for c in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for a in algos:
            cells = []
            for key, _, nd in cols:
                vals = np.array([r[key] for r in group if r["algo"] == a])
                vals = vals[np.isfinite(vals)]
                cells.append(_ci(iqm_ci({a: vals}, reps)[a]["iqm"], nd) if len(vals) else "—")
            lines.append(f"| {a.upper()} ({sum(r['algo'] == a for r in group)} semillas) | " + " | ".join(cells) + " |")
        if pid:
            lines.append("| PID | " + " | ".join(f"{pid[k]:.{nd}f}" if np.isfinite(pid[k]) else "—" for k, _, nd in cols) + " |")
        lines.append("")
    md_path = docs_dir / "empujon.md"
    md_path.write_text("\n".join(lines))
    return [md_path, csv_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba de empujón (E4)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--out", default=str(DEFAULT_OUT / "push"))
    parser.add_argument("--config", default=str(EVAL_CONFIG))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pid", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    cfg = load_eval_config(args.config)
    if not args.summary_only:
        targets: list[Any] = ([] if args.no_pid else ["pid"]) + finished_runs(Path(args.runs_dir))
        print(f"Prueba de empujón de {len(targets)} políticas con {args.workers} procesos...", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(push_test_run, t, args.out, cfg, args.force) for t in targets]
            for t, fut in zip(targets, futures):
                print(f"  {Path(str(t)).name}: {fut.result()}", flush=True)
    for p in summarize(Path(args.out)):
        print(p)


if __name__ == "__main__":
    main()
