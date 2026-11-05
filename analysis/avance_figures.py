"""Figuras y tablas del avance del 2026-11-02.

Uso:
    python -m analysis.avance_figures pid     # respuesta del PID a un empujón + tabla por nivel
    python -m analysis.avance_figures smoke   # curvas de la prueba de humo (results/smoke*)
Las figuras PNG van a docs/img/ (se versionan); tablas y videos a results/avance/ (no se versionan).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from baselines.pid_controller import PIDController
from envs import BalanceBotEnv
from envs.balance_bot_env import REPO_ROOT

OUT = REPO_ROOT / "results" / "avance"
FIG = REPO_ROOT / "docs" / "img"
# Paleta categórica validada (dataviz, modo claro): azul, naranja, aqua, en orden fijo.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pid_push(force: float = 6.0, push_time: float = 2.0, seed: int = 0) -> Path:
    """θ, v y acción del PID ante un empujón hacia adelante en el nivel 0 (paneles con el mismo eje de tiempo)."""
    env = BalanceBotEnv(level=0)
    pid = PIDController.from_env(env)
    obs, _ = env.reset(seed=seed)
    env.perturbations.schedule(push_time, force, 0.0)
    t, theta, v, act = [], [], [], []
    for _ in range(600):
        a, _ = pid.predict(obs)
        obs, _, terminated, _, info = env.step(a)
        t.append(env.data.time)
        theta.append(np.degrees(info["pitch"]))
        v.append(info["v"])
        act.append(a.mean())
        if terminated:
            break

    fig, axes = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True, facecolor=SURFACE)
    series = [(theta, "Pitch θ (°)"), (v, "Velocidad v (m/s)"), (act, "Acción media (−1 a 1)")]
    for ax, (y, label) in zip(axes, series):
        _style(ax)
        ax.axvspan(push_time, push_time + env.perturbations.duration, color=GRID, alpha=0.9, lw=0)
        ax.plot(t, y, color=BLUE, lw=2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_ylabel(label, color=INK, fontsize=10)
    axes[0].annotate(
        f"empujón de {force:.0f} N, 0.1 s", xy=(push_time + 0.1, max(theta)), xytext=(push_time + 0.6, max(theta)),
        color=INK_2, fontsize=9, va="center", arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.8},
    )
    axes[-1].set_xlabel("Tiempo (s)", color=INK, fontsize=10)
    fig.suptitle("PID en cascada: recuperación ante un empujón (nivel 0)", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    path = FIG / "pid_empujon.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def pid_levels(seeds: int = 20) -> str:
    """Tabla markdown: caídas, retorno y error de seguimiento del PID por nivel."""
    rows = ["| Nivel | Caídas | Retorno medio | RMSE v (m/s) | RMSE ψ̇ (rad/s) |", "|---|---|---|---|---|"]
    for level in range(4):
        env = BalanceBotEnv(level=level)
        pid = PIDController.from_env(env)
        falls, rets, ev, ey = 0, [], [], []
        for s in range(seeds):
            obs, _ = env.reset(seed=s)
            pid.reset()
            ret = 0.0
            while True:
                a, _ = pid.predict(obs)
                obs, r, terminated, truncated, info = env.step(a)
                ret += r
                ev.append(info["v"] - info["command"][0])
                ey.append(info["yaw_rate"] - info["command"][1])
                if terminated or truncated:
                    break
            falls += terminated
            rets.append(ret)
        rmse = lambda e: float(np.sqrt(np.mean(np.square(e))))  # noqa: E731
        rows.append(f"| {level} | {falls}/{seeds} | {np.mean(rets):.1f} | {rmse(ev):.3f} | {rmse(ey):.3f} |")
    table = "\n".join(rows)
    (OUT / "pid_niveles.md").write_text(table + "\n")
    return table


def smoke_curves() -> Path:
    """Retorno de evaluación vs pasos: un panel por algoritmo; semillas y variante normalizada como series."""
    runs = []
    for d in sorted(glob.glob(str(REPO_ROOT / "results" / "smoke*" / "*"))):
        d = Path(d)
        if not (d / "evaluations.npz").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        e = np.load(d / "evaluations.npz")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        runs.append(
            {
                "algo": cfg["algo"]["algo"].upper(),
                "seed": cfg["seed"],
                "norm": bool(cfg["common"]["normalize"]),
                "steps": e["timesteps"],
                "ret": e["results"].mean(axis=1),
                "sps": meta.get("steps_per_s"),
            }
        )
    algos = sorted({r["algo"] for r in runs})
    fig, axes = plt.subplots(1, len(algos), figsize=(5.2 * len(algos), 4), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, algo in zip(axes, algos):
        _style(ax)
        mine = sorted((r for r in runs if r["algo"] == algo), key=lambda r: (r["norm"], r["seed"]))
        colors = [BLUE, ORANGE, AQUA]
        for r, c in zip(mine, colors):
            label = f"semilla {r['seed']}" + (" · normalizado" if r["norm"] else "")
            ax.plot(r["steps"] / 1000, r["ret"], color=c, lw=2, marker="o", ms=4, label=label)
        ax.axhline(1000, color=INK_2, lw=0.8, ls="--")
        ax.set_title(algo, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Pasos de entorno (miles)", color=INK, fontsize=10)
    axes[0].set_ylabel("Retorno de evaluación (máx. 1000)", color=INK, fontsize=10)
    # Una sola leyenda: el color es la misma corrida (semilla / normalización) en todos los paneles.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle("Prueba de humo, nivel 0: valores por defecto de SB3", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    # Un archivo por grupo de algoritmos (p. ej. prueba_humo_sac_td3.png): cada quien corre los suyos y no se pisan.
    stem = "prueba_humo_" + "_".join(a.lower() for a in algos)
    path = FIG / f"{stem}.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    lines = ["| Algoritmo | Semilla | Normalizado | Retorno final | Mejor retorno | Pasos/s |", "|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (r["algo"], r["norm"], r["seed"])):
        lines.append(f"| {r['algo']} | {r['seed']} | {'sí' if r['norm'] else 'no'} | {r['ret'][-1]:.0f} | "
                     f"{r['ret'].max():.0f} | {r['sps'] if r['sps'] else 'en curso'} |")
    (OUT / f"{stem}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["pid", "smoke"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.what == "pid":
        print(pid_push())
        print(pid_levels())
    else:
        print(smoke_curves())


if __name__ == "__main__":
    main()
