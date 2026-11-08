"""Análisis de las corridas finales (E12 → E13; especificación §7.3–§7.5 y §8).

Uso:
    python -m analysis.finales [--runs-dir results/final] [--last-k 3] [--reps 2000]

Lee cada corrida terminada de `--runs-dir` (las que dejan `lab.semillas` / `training.train`: config.yaml,
meta.json, evaluations.npz, diagnostics.npz y TensorBoard) y, por nivel de entrenamiento (2: §7.1; 3: E5), deja:

- `docs/finales/level<n>.md`: corridas por algoritmo y máquina, desempeño final con IQM y media e intervalos
  bootstrap al 95 % (§7.5; IQM de `rliable`), eficiencia de muestras, suavidad y sobreestimación de Q (§8, H1).
- `docs/finales/level<n>.csv`: una fila por corrida.
- `docs/img/finales_level<n>_pasos.png` (E1): retorno de evaluación vs pasos, IQM entre semillas con IC 95 %.
- `docs/img/finales_level<n>_reloj.png` (E2): retorno vs horas de reloj, un panel por máquina (el tiempo de
  reloj solo se compara dentro de una misma máquina).
- `docs/img/finales_level<n>_q.png` (H1): sobreestimación de Q vs pasos (DDPG, TD3, SAC).

E3, E4 y E6 (evaluación en los niveles 0–3, empujón y PID) los cubre `evaluation/` (E11), no este script.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rliable import metrics
from scipy.stats import trim_mean

from envs.balance_bot_env import REPO_ROOT

DEFAULT_RUNS = REPO_ROOT / "results" / "final"
DOCS = REPO_ROOT / "docs" / "finales"
FIG = REPO_ROOT / "docs" / "img"
ALGO_ORDER = ("ddpg", "td3", "sac", "ppo")
SAMPLE_EFFICIENCY_FRACTION = 0.9  # §7.4: pasos para alcanzar el 90 % del retorno final

# Paleta categórica de referencia (dataviz, modo claro), en orden fijo y con el color atado al algoritmo.
COLORS = {"ddpg": "#2a78d6", "td3": "#eb6834", "sac": "#1baf7a", "ppo": "#eda100"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


@dataclass
class Run:
    path: Path
    algo: str
    level: int
    seed: int
    hostname: str
    duration_s: float
    steps_per_s: float
    commit: str | None
    timesteps: np.ndarray  # pasos de cada evaluación
    returns: np.ndarray  # retorno medio de cada evaluación
    wall_h: np.ndarray | None = None  # horas desde el inicio del entrenamiento en cada evaluación (TensorBoard)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


def _wall_hours(run_dir: Path, duration_s: float, n_evals: int) -> np.ndarray | None:
    """Horas de reloj de cada evaluación, con el cero en el inicio del entrenamiento.

    TensorBoard guarda la hora de cada `eval/mean_reward`. El cero es la hora del último evento menos la duración
    medida por `train()`: no depende de la zona horaria de la máquina (Colab está en UTC).
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = []
    for tb in sorted((run_dir / "tb").glob("*")):
        acc = EventAccumulator(str(tb), size_guidance={"scalars": 0}).Reload()
        for tag in acc.Tags()["scalars"]:
            events += [(e.wall_time, tag, e.step) for e in acc.Scalars(tag)]
    evals = sorted(w for w, tag, _ in events if tag == "eval/mean_reward")
    if not events or len(evals) < n_evals:
        return None
    t0 = max(w for w, _, _ in events) - duration_s
    return (np.asarray(evals[:n_evals]) - t0) / 3600


def load_runs(runs_dir: Path) -> list[Run]:
    """Corridas terminadas (con meta.json y evaluations.npz) de `runs_dir`; algoritmo, nivel y semilla salen de config.yaml."""
    runs = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        if not ((run_dir / "meta.json").exists() and (run_dir / "evaluations.npz").exists()):
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        meta = json.loads((run_dir / "meta.json").read_text())
        with np.load(run_dir / "evaluations.npz") as ev:
            timesteps, returns = ev["timesteps"], ev["results"].mean(axis=1)
        diag = {}
        if (run_dir / "diagnostics.npz").exists():
            with np.load(run_dir / "diagnostics.npz") as d:
                diag = {k: d[k] for k in d.files}
        runs.append(Run(
            path=run_dir, algo=cfg["algo"]["algo"], level=int(cfg["env"]["level"]), seed=int(cfg["seed"]),
            hostname=meta.get("machine", {}).get("hostname", "?"), duration_s=float(meta["duration_s"]),
            steps_per_s=float(meta["steps_per_s"]), commit=(meta.get("git") or {}).get("commit"),
            timesteps=timesteps, returns=returns, wall_h=_wall_hours(run_dir, float(meta["duration_s"]), len(timesteps)),
            diagnostics=diag,
        ))
    return runs


def final_return(run: Run, last_k: int) -> float:
    """Retorno final de una corrida: media de sus últimas `last_k` evaluaciones (igual que el objetivo de §6)."""
    return float(run.returns[-last_k:].mean())


def steps_to_fraction(run: Run, last_k: int, fraction: float = SAMPLE_EFFICIENCY_FRACTION) -> float:
    """§7.4: primeros pasos en que la evaluación alcanza `fraction` del retorno final; NaN si el final es ≤ 0."""
    target = fraction * final_return(run, last_k)
    if target <= 0:
        return float("nan")
    hit = np.flatnonzero(run.returns >= target)
    return float(run.timesteps[hit[0]]) if hit.size else float("nan")


def bootstrap_ci(x: np.ndarray, stat, reps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estadístico y su IC 95 % por bootstrap percentil, remuestreando corridas (primer eje) con reemplazo.

    Es el método por defecto de `rliable.library.get_interval_estimates` (percentil, 95 %) con una sola tarea, donde
    el bootstrap estratificado es remuestrear corridas. Se implementa aquí porque `rliable.library` depende de
    `arch`, que en este entorno no es compatible con pandas 3; el IQM sí sale de `rliable.metrics`.
    En curvas se remuestrean corridas completas: los puntos de una misma corrida van juntos.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.array([stat(x[idx]) for idx in rng.integers(0, len(x), size=(reps, len(x)))])
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return np.asarray(stat(x)), low, high


def iqm_ci(values_by_algo: dict[str, np.ndarray], reps: int, seed: int = 0) -> dict[str, dict[str, tuple[float, float, float]]]:
    """IQM y media con IC 95 % (§7.5). `values_by_algo[a]` tiene una entrada por corrida."""
    out = {}
    for algo, values in values_by_algo.items():
        if len(values):
            scores = np.asarray(values, dtype=float).reshape(-1, 1)  # rliable: corridas × tareas (una tarea)
            out[algo] = {name: tuple(float(v) for v in bootstrap_ci(scores, stat, reps, seed))
                         for name, stat in (("iqm", metrics.aggregate_iqm), ("mean", metrics.aggregate_mean))}
    return out


def iqm_curve(curves: np.ndarray, reps: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """IQM entre corridas en cada punto de una curva (forma: corridas × puntos) y su IC 95 % (2 × puntos)."""
    point, low, high = bootstrap_ci(curves, lambda c: trim_mean(c, 0.25, axis=0), reps, seed)  # = aggregate_iqm por punto
    return point, np.stack([low, high])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _algos(runs: list[Run]) -> list[str]:
    present = {r.algo for r in runs}
    return [a for a in ALGO_ORDER if a in present] + sorted(present - set(ALGO_ORDER))


def _aligned(runs: list[Run], key: str = "returns") -> tuple[np.ndarray, np.ndarray]:
    """Curvas de todas las corridas recortadas al largo común (todas evalúan cada `eval.freq` pasos)."""
    n = min(len(getattr(r, key) if key == "returns" else r.diagnostics[key]) for r in runs)
    x = runs[0].timesteps[:n] if key == "returns" else runs[0].diagnostics["timesteps"][:n]
    y = np.stack([(r.returns if key == "returns" else r.diagnostics[key])[:n] for r in runs])
    return x, y


def plot_steps(runs: list[Run], level: int, reps: int, path: Path, max_return: float | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=SURFACE)
    _style(ax)
    for algo in _algos(runs):
        mine = [r for r in runs if r.algo == algo]
        x, y = _aligned(mine)
        point, ci = iqm_curve(y, reps)
        ax.fill_between(x / 1000, ci[0], ci[1], color=COLORS.get(algo, INK_2), alpha=0.18, lw=0)
        ax.plot(x / 1000, point, color=COLORS.get(algo, INK_2), lw=2, label=f"{algo.upper()} ({len(mine)} semillas)")
    if max_return is not None:
        ax.axhline(max_return, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel("Pasos de entorno (miles)", color=INK, fontsize=10)
    ax.set_ylabel("Retorno de evaluación (IQM, IC 95 %)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    fig.suptitle(f"E1: curvas de aprendizaje por pasos, entrenamiento en el nivel {level}", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_wall_clock(runs: list[Run], level: int, path: Path) -> None:
    timed = [r for r in runs if r.wall_h is not None]
    hosts = sorted({r.hostname for r in timed})
    if not hosts:
        return
    fig, axes = plt.subplots(1, len(hosts), figsize=(max(8, 6 * len(hosts)), 4.4), sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, host in zip(axes[0], hosts):
        _style(ax)
        for algo in _algos(timed):
            for i, r in enumerate(x for x in timed if x.algo == algo and x.hostname == host):
                ax.plot(r.wall_h, r.returns, color=COLORS.get(algo, INK_2), lw=1.2, alpha=0.8, label=algo.upper() if i == 0 else None)
        ax.set_title(f"Máquina: {host}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Horas de reloj", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    axes[0][0].set_ylabel("Retorno de evaluación (una línea por semilla)", color=INK, fontsize=10)
    fig.suptitle(f"E2: retorno vs tiempo de reloj, nivel {level}", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_q(runs: list[Run], level: int, reps: int, path: Path) -> bool:
    with_q = [r for r in runs if "q_overestimation" in r.diagnostics and np.isfinite(r.diagnostics["q_overestimation"]).all()]
    if not with_q:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.4), facecolor=SURFACE)
    _style(ax)
    for algo in _algos(with_q):
        mine = [r for r in with_q if r.algo == algo]
        x, y = _aligned(mine, "q_overestimation")
        point, ci = iqm_curve(y, reps)
        ax.fill_between(x / 1000, ci[0], ci[1], color=COLORS.get(algo, INK_2), alpha=0.18, lw=0)
        ax.plot(x / 1000, point, color=COLORS.get(algo, INK_2), lw=2, marker="o", ms=4, label=f"{algo.upper()} ({len(mine)} semillas)")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xlabel("Pasos de entorno (miles)", color=INK, fontsize=10)
    ax.set_ylabel("Q(s, π(s)) − retorno real (IQM, IC 95 %)", color=INK, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"H1: sobreestimación de Q, entrenamiento en el nivel {level}", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def _ci(t: tuple[float, float, float], nd: int = 1) -> str:
    return f"{t[0]:.{nd}f} [{t[1]:.{nd}f}, {t[2]:.{nd}f}]"


def _diag_final(r: Run, key: str, last_k: int) -> float:
    v = r.diagnostics.get(key)
    return float(np.nanmean(v[-last_k:])) if v is not None and len(v) and np.isfinite(v[-last_k:]).any() else float("nan")


def render_markdown(runs: list[Run], level: int, last_k: int, reps: int, figures: dict[str, str]) -> str:
    algos = _algos(runs)
    by_algo = {a: [r for r in runs if r.algo == a] for a in algos}
    out = [f"# Corridas finales: entrenamiento en el nivel {level}", "",
           f"Generado con `python -m analysis.finales`. Retorno final de una corrida = media de sus últimas {last_k} "
           f"evaluaciones. IQM (`rliable.metrics`) y media con IC 95 % por bootstrap percentil sobre corridas ({reps} remuestreos, §7.5).", ""]

    out += ["## Corridas", "", "| Algoritmo | Semillas | Máquinas | Pasos por corrida | Pasos/s (mediana) | Commit |", "|---|---|---|---|---|---|"]
    for a in algos:
        rs = by_algo[a]
        hosts = defaultdict(list)
        for r in rs:
            hosts[r.hostname].append(r.seed)
        out.append(f"| {a.upper()} | {len(rs)}: {sorted(r.seed for r in rs)} | "
                   + "; ".join(f"{h} {sorted(s)}" for h, s in sorted(hosts.items()))
                   + f" | {int(rs[0].timesteps[-1])} | {np.median([r.steps_per_s for r in rs]):.0f} | "
                   + ", ".join(sorted({(r.commit or '?')[:7] for r in rs})) + " |")
    if any(len(by_algo[a]) < 5 for a in algos):
        out += ["", "**Aviso:** algún algoritmo tiene menos de 5 semillas, el mínimo de §7.1."]
    repeated = {f"{a.upper()} {s}" for a in algos for s in {r.seed for r in by_algo[a]} if sum(r.seed == s for r in by_algo[a]) > 1}
    if repeated:
        out += ["", f"**Aviso:** semillas repetidas ({', '.join(sorted(repeated))}): ¿la misma semilla se corrió en dos máquinas? "
                "Cada semilla debe aparecer una sola vez."]
    out.append("")

    finals = {a: np.array([final_return(r, last_k) for r in by_algo[a]]) for a in algos}
    stats = iqm_ci(finals, reps)
    efficiency = {a: np.array([steps_to_fraction(r, last_k) for r in by_algo[a]]) for a in algos}
    out += ["## Desempeño final y eficiencia de muestras (E1, §7.4)", "",
            "| Algoritmo | IQM del retorno final | Media del retorno final | Peor / mejor semilla | Pasos hasta el 90 % del final (mediana) |",
            "|---|---|---|---|---|"]
    for a in algos:
        eff = efficiency[a][np.isfinite(efficiency[a])]
        out.append(f"| {a.upper()} | {_ci(stats[a]['iqm'])} | {_ci(stats[a]['mean'])} | {finals[a].min():.1f} / {finals[a].max():.1f} | "
                   + (f"{np.median(eff) / 1000:.0f}k" if eff.size else "–") + " |")
    out.append("")

    if any(r.diagnostics for r in runs):
        out += ["## Diagnósticos (§8)", "",
                f"Valores al final del entrenamiento (media de los últimos {last_k} diagnósticos de cada corrida), IQM entre semillas con IC 95 %.", "",
                "| Algoritmo | Sobreestimación de Q | ‖aₜ − aₜ₋₁‖ medio | Energía > corte (fracción) |", "|---|---|---|---|"]
        for a in algos:
            cells = []
            for key, nd in (("q_overestimation", 1), ("action_smoothness", 3), ("high_freq_energy", 3)):
                vals = np.array([_diag_final(r, key, last_k) for r in by_algo[a]])
                vals = vals[np.isfinite(vals)]
                cells.append(_ci(iqm_ci({a: vals}, reps)[a]["iqm"], nd) if vals.size else "–")
            out.append(f"| {a.upper()} | " + " | ".join(cells) + " |")
        out.append("")

    notes = {"reloj": "El tiempo de reloj depende del hardware: solo se comparan algoritmos dentro de un mismo panel (máquina)."}
    for title, key in (("E1: curvas por pasos", "pasos"), ("E2: curvas por tiempo de reloj", "reloj"), ("H1: sobreestimación de Q", "q")):
        if key in figures:
            out += [f"## {title}", ""] + ([notes[key], ""] if key in notes else []) + [f"![{title}](../img/{figures[key]})", ""]

    out += ["## Todas las corridas", "", "| Algoritmo | Semilla | Máquina | Retorno final | Mejor eval. | Pasos al 90 % | Horas |", "|---|---|---|---|---|---|---|"]
    for a in algos:
        for r in sorted(by_algo[a], key=lambda r: r.seed):
            eff = steps_to_fraction(r, last_k)
            out.append(f"| {a.upper()} | {r.seed} | {r.hostname} | {final_return(r, last_k):.1f} | {r.returns.max():.1f} | "
                       f"{'–' if np.isnan(eff) else f'{eff / 1000:.0f}k'} | {r.duration_s / 3600:.1f} |")
    return "\n".join(out) + "\n"


def write_csv(runs: list[Run], last_k: int, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo", "nivel", "semilla", "maquina", "retorno_final", "mejor_eval", "pasos_al_90", "horas", "pasos_por_s",
                    "q_overestimation_final", "suavidad_final", "energia_alta_final", "commit", "carpeta"])
        for r in sorted(runs, key=lambda r: (r.algo, r.seed)):
            w.writerow([r.algo, r.level, r.seed, r.hostname, round(final_return(r, last_k), 2), round(float(r.returns.max()), 2),
                        steps_to_fraction(r, last_k), round(r.duration_s / 3600, 2), r.steps_per_s,
                        *(round(_diag_final(r, k, last_k), 4) for k in ("q_overestimation", "action_smoothness", "high_freq_energy")),
                        r.commit, r.path.name])


def analyze(runs_dir: Path, last_k: int = 3, reps: int = 2000, out_dir: Path = DOCS, fig_dir: Path = FIG) -> list[Path]:
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No hay corridas terminadas en {runs_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for level in sorted({r.level for r in runs}):
        mine = [r for r in runs if r.level == level]
        max_return = float(yaml.safe_load((mine[0].path / "config.yaml").read_text())["env"]["max_episode_steps"])
        figures = {"pasos": f"finales_level{level}_pasos.png", "reloj": f"finales_level{level}_reloj.png"}
        plot_steps(mine, level, reps, fig_dir / figures["pasos"], max_return)
        plot_wall_clock(mine, level, fig_dir / figures["reloj"])
        if plot_q(mine, level, reps, fig_dir / f"finales_level{level}_q.png"):
            figures["q"] = f"finales_level{level}_q.png"
        write_csv(mine, last_k, out_dir / f"level{level}.csv")
        md = out_dir / f"level{level}.md"
        md.write_text(render_markdown(mine, level, last_k, reps, figures), encoding="utf-8")
        written.append(md)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza las corridas finales (E13)")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    parser.add_argument("--last-k", type=int, default=3, help="evaluaciones finales que se promedian (igual que el objetivo de §6)")
    parser.add_argument("--reps", type=int, default=2000, help="remuestreos bootstrap")
    parser.add_argument("--out-dir", default=str(DOCS))
    parser.add_argument("--fig-dir", default=str(FIG))
    args = parser.parse_args()
    for md in analyze(Path(args.runs_dir), args.last_k, args.reps, Path(args.out_dir), Path(args.fig_dir)):
        print(md)


if __name__ == "__main__":
    main()
