"""Resumen de los estudios de afinamiento de Optuna (E10, especificación §6).

Uso:
    python -m analysis.resumen_afinamiento                # todos los estudios de <storage_dir> (tune.yaml)
    python -m analysis.resumen_afinamiento --algo ddpg    # solo uno

Por cada estudio `<algo>_level<n>` deja (se versionan, para que la evidencia de E10 quede en el repo aunque el
`.db` y `results/` no se suban):
- `docs/afinamiento/<estudio>.md`: protocolo, conteo, tiempos, mejores pruebas, configuración elegida y tabla
  de todas las pruebas.
- `docs/afinamiento/<estudio>.csv`: una fila por prueba.
- `docs/img/afinamiento_<estudio>.png`: valor objetivo por prueba y curvas de evaluación de las 3 mejores.

Cada quien corre el script con sus estudios (Mario: DDPG y PPO; Christian: TD3 y SAC); los archivos no se pisan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

from envs.balance_bot_env import REPO_ROOT
from training.train import load_yaml
from training.tune import TUNE_CONFIG

DOCS = REPO_ROOT / "docs" / "afinamiento"
FIG = REPO_ROOT / "docs" / "img"
INTERRUPTED_ATTR = "interrumpidas"  # lo escribe lab.afinar al marcar pruebas cortadas
TOP_K = 5
CURVES_K = 3  # la paleta categórica validada tiene 3 tonos
EDGE_FRACTION = 0.05  # un valor a menos del 5 % del borde del rango (en escala log si aplica) se marca

# Misma paleta que analysis/avance_figures.py (validada con dataviz, modo claro).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def trial_rows(study: optuna.Study, runs_root: Path) -> list[dict[str, Any]]:
    """Una fila por prueba con su estado, valor, pasos alcanzados, duración y parámetros."""
    interrupted = set(study.user_attrs.get(INTERRUPTED_ATTR, []))
    rows = []
    for t in study.get_trials(deepcopy=False):
        state = "INTERRUMPIDA" if t.number in interrupted else t.state.name
        steps, last_value = (max(t.intermediate_values), t.intermediate_values[max(t.intermediate_values)]) if t.intermediate_values else (0, None)
        run_dir = Path(t.user_attrs["run_dirs"][0]) if t.user_attrs.get("run_dirs") else runs_root / f"trial{t.number:03d}_seed0"
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        evals = np.load(run_dir / "evaluations.npz") if (run_dir / "evaluations.npz").exists() else None
        rows.append({
            "numero": t.number,
            "estado": state,
            "valor": t.value if t.state == optuna.trial.TrialState.COMPLETE else None,
            "ultimo_valor_intermedio": last_value,
            "pasos": steps,
            "eval_final": float(evals["results"][-1].mean()) if evals is not None and len(evals["results"]) else None,
            "eval_mejor": float(evals["results"].mean(axis=1).max()) if evals is not None and len(evals["results"]) else None,
            "minutos": (t.datetime_complete - t.datetime_start).total_seconds() / 60 if t.datetime_complete and t.datetime_start else None,
            "pasos_por_s": meta.get("steps_per_s"),
            "inicio": t.datetime_start,
            "fin": t.datetime_complete,
            "commit": (meta.get("git") or {}).get("commit"),
            "dirty": (meta.get("git") or {}).get("dirty"),
            "versions": meta.get("versions"),
            "hostname": (meta.get("machine") or {}).get("hostname"),
            "params": dict(t.params),
            "run_dir": run_dir,
        })
    return rows


def near_edge(value: float, spec: dict[str, Any]) -> str:
    """'bajo'/'alto' si el valor quedó cerca de un borde del rango de búsqueda (hay que ampliar el rango)."""
    if spec["type"] == "categorical":
        choices = spec["choices"]
        return "bajo" if value == choices[0] else "alto" if value == choices[-1] else ""
    low, high = float(spec["low"]), float(spec["high"])
    f = (lambda x: math.log(x)) if spec.get("log") else (lambda x: x)
    pos = (f(value) - f(low)) / (f(high) - f(low))
    return "bajo" if pos < EDGE_FRACTION else "alto" if pos > 1 - EDGE_FRACTION else ""


def _fmt(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if isinstance(x, float):  # retornos con `nd` decimales; hiperparámetros con 3 cifras significativas
        return f"{x:.{nd}f}" if abs(x) >= 100 else f"{x:.3g}"
    return str(x)


def _current_value(hyperparams: dict[str, Any], dotted: str) -> Any:
    node: Any = hyperparams
    for part in dotted.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def render_markdown(name: str, study: optuna.Study, rows: list[dict[str, Any]], algo_cfg: dict[str, Any], tune_cfg: dict[str, Any], figure: str | None) -> str:
    protocol = study.user_attrs.get("protocol", {})
    counts = {s: sum(r["estado"] == s for r in rows) for s in ("COMPLETE", "PRUNED", "FAIL", "INTERRUMPIDA", "RUNNING")}
    valid = [r for r in rows if r["estado"] in ("COMPLETE", "PRUNED", "FAIL")]
    complete = sorted((r for r in rows if r["estado"] == "COMPLETE"), key=lambda r: -r["valor"])
    starts = [r["inicio"] for r in rows if r["inicio"]]
    ends = [r["fin"] for r in rows if r["fin"]]
    cpu_h = sum(r["minutos"] or 0 for r in valid) / 60
    steps = sum(r["pasos"] for r in valid)
    commits = sorted({(r["commit"] or "?")[:7] + (" (con cambios sin commit)" if r["dirty"] else "") for r in valid if r["commit"]})
    versions = next((r["versions"] for r in valid if r["versions"]), None)
    hosts = sorted({r["hostname"] for r in valid if r["hostname"]})
    space = algo_cfg.get("search_space") or {}

    out = [f"# Afinamiento: {name}", "",
           f"Generado el {datetime.now():%Y-%m-%d %H:%M} con `python -m analysis.resumen_afinamiento`. "
           "Fuente: el estudio de Optuna (`.db`) y las corridas de cada prueba (no se versionan).", ""]
    out += ["## Protocolo", "",
            f"- Nivel {protocol.get('env', {}).get('level', '?')}, {protocol.get('trial_timesteps', '?')} pasos por prueba, semillas {protocol.get('seeds', '?')}.",
            f"- Objetivo: media de las últimas {protocol.get('objective', {}).get('last_k_evals', '?')} evaluaciones (§6).",
            f"- Muestreador {tune_cfg['sampler']['type'].upper()} ({tune_cfg['sampler']['n_startup_trials']} pruebas al azar al inicio); "
            f"podador {tune_cfg['pruner']['type']} (sin poda en las primeras {tune_cfg['pruner']['n_startup_trials']} completas ni antes de "
            f"{tune_cfg['pruner']['n_warmup_steps']} pasos). Según `tune.yaml` al generar este resumen.",
            f"- Código: commit {', '.join(commits) or '?'}. Máquina: {', '.join(hosts) or '?'}.",
            f"- Versiones: {', '.join(f'{k} {v}' for k, v in versions.items()) if versions else '?'}.", ""]
    out += ["## Resultado", "",
            f"- **{len(valid)} pruebas válidas**: {counts['COMPLETE']} completas, {counts['PRUNED']} podadas, {counts['FAIL']} fallidas (divergencia)."
            + (f" Además, {counts['INTERRUMPIDA']} interrumpidas (corte del proceso; no cuentan)." if counts["INTERRUMPIDA"] else "")
            + (f" **{counts['RUNNING']} siguen corriendo: el estudio no ha terminado.**" if counts["RUNNING"] else ""),
            f"- Duración: {min(starts):%Y-%m-%d %H:%M} a {max(ends):%H:%M} ({(max(ends) - min(starts)).total_seconds() / 3600:.1f} h de reloj, "
            f"{cpu_h:.1f} h de CPU sumando procesos). {steps / 1e6:.2f} M pasos de entorno en total." if starts and ends else "- Duración: –",
            ""]
    if complete:
        best = complete[0]
        out += [f"### Mejor prueba: #{best['numero']} (objetivo {best['valor']:.1f})", "",
                f"Evaluación final {_fmt(best['eval_final'])}, mejor evaluación {_fmt(best['eval_mejor'])}, {_fmt(best['pasos_por_s'], 0)} pasos/s.", "",
                "| Hiperparámetro | Rango de búsqueda | En el YAML al generar | Mejor prueba | ¿Cerca del borde? |", "|---|---|---|---|---|"]
        for key, spec in space.items():
            rng = str(spec["choices"]) if spec["type"] == "categorical" else f"{spec['low']} – {spec['high']}" + (" (log)" if spec.get("log") else "")
            value = best["params"].get(key)
            out.append(f"| `{key}` | {rng} | {_fmt(_current_value(algo_cfg.get('hyperparams') or {}, key))} | {_fmt(value)} | "
                       f"{near_edge(value, spec) if value is not None else ''} |")
        out += ["", f"### Las {min(TOP_K, len(complete))} mejores", "",
                "| # | Objetivo | Eval. final | Mejor eval. | Min | " + " | ".join(f"`{k}`" for k in space) + " |",
                "|---" * (5 + len(space)) + "|"]
        for r in complete[:TOP_K]:
            out.append(f"| {r['numero']} | {r['valor']:.1f} | {_fmt(r['eval_final'])} | {_fmt(r['eval_mejor'])} | {_fmt(r['minutos'], 0)} | "
                       + " | ".join(_fmt(r["params"].get(k)) for k in space) + " |")
        out += ["", "Cada prueba usa una sola semilla: diferencias pequeñas entre las mejores pueden ser ruido de la semilla.", ""]
    if figure:
        out += [f"![Afinamiento de {name}](../img/{figure})", ""]
    out += ["## Todas las pruebas", "",
            "Para las podadas, *objetivo* es el último valor intermedio (media de las últimas evaluaciones hasta ese paso).", "",
            "| # | Estado | Objetivo | Pasos | Min | " + " | ".join(f"`{k}`" for k in space) + " |", "|---" * (5 + len(space)) + "|"]
    state_es = {"COMPLETE": "completa", "PRUNED": "podada", "FAIL": "fallida", "INTERRUMPIDA": "interrumpida", "RUNNING": "corriendo"}
    for r in rows:
        value = r["valor"] if r["valor"] is not None else r["ultimo_valor_intermedio"]
        out.append(f"| {r['numero']} | {state_es.get(r['estado'], r['estado'])} | {_fmt(value)} | {r['pasos'] or '–'} | {_fmt(r['minutos'], 0)} | "
                   + " | ".join(_fmt(r["params"].get(k)) for k in space) + " |")
    return "\n".join(out) + "\n"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for r in rows for k in r["params"]})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["numero", "estado", "valor", "ultimo_valor_intermedio", "pasos", "eval_final", "eval_mejor", "minutos", "pasos_por_s", *keys])
        for r in rows:
            w.writerow([r["numero"], r["estado"], r["valor"], r["ultimo_valor_intermedio"], r["pasos"], r["eval_final"], r["eval_mejor"],
                        None if r["minutos"] is None else round(r["minutos"], 1), r["pasos_por_s"], *(r["params"].get(k) for k in keys)])


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_study(name: str, rows: list[dict[str, Any]], max_value: float | None, path: Path) -> None:
    """Izquierda: objetivo por prueba (completas y podadas) y mejor hasta el momento. Derecha: curvas de las 3 mejores."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=SURFACE)
    for ax in (ax1, ax2):
        _style(ax)

    floor = -200.0  # recorte del eje: las podadas que divergen (hasta −1000) aplastarían el resto
    for state, color, label, filled in (("COMPLETE", BLUE, "completa", True), ("PRUNED", ORANGE, "podada (último valor intermedio)", False)):
        pts = [(r["numero"], r["valor"] if r["valor"] is not None else r["ultimo_valor_intermedio"]) for r in rows if r["estado"] == state]
        pts = [(n, v) for n, v in pts if v is not None]
        inside = [(n, v) for n, v in pts if v >= floor]
        below = [(n, floor) for n, v in pts if v < floor]
        if inside:
            ax1.scatter(*zip(*inside), s=36, color=color if filled else SURFACE, edgecolors=color, linewidths=2, label=label, zorder=3)
        if below:
            ax1.scatter(*zip(*below), s=40, marker="v", color=color, zorder=3, label=f"{label}, < {floor:.0f}")
    best_so_far, best = [], -math.inf
    for r in rows:
        if r["estado"] == "COMPLETE":
            best = max(best, r["valor"])
        best_so_far.append((r["numero"], best if best > -math.inf else np.nan))
    ax1.step(*zip(*best_so_far), where="post", color=INK_2, lw=1.5, label="mejor completa hasta ese momento", zorder=2)
    ax1.set_ylim(bottom=floor - 40)
    ax1.set_xlabel("Número de prueba", color=INK, fontsize=10)
    ax1.set_ylabel("Valor objetivo", color=INK, fontsize=10)
    ax1.set_title("Objetivo por prueba", color=INK, fontsize=11, loc="left")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)  # debajo: no tapa puntos

    top = sorted((r for r in rows if r["estado"] == "COMPLETE"), key=lambda r: -r["valor"])[:CURVES_K]
    for r, color in zip(top, (BLUE, ORANGE, AQUA)):
        npz = r["run_dir"] / "evaluations.npz"
        if not npz.exists():
            continue
        ev = np.load(npz)
        x, y = ev["timesteps"] / 1000, ev["results"].mean(axis=1)
        ax2.plot(x, y, color=color, lw=2, marker="o", ms=4, label=f"#{r['numero']} (objetivo {r['valor']:.0f})")
    if max_value is not None:
        ax2.axhline(max_value, color=INK_2, lw=0.8, ls="--")
    ax2.set_xlabel("Pasos de entorno (miles)", color=INK, fontsize=10)
    ax2.set_ylabel("Retorno de evaluación", color=INK, fontsize=10)
    ax2.set_title(f"Curvas de las {len(top)} mejores", color=INK, fontsize=11, loc="left")
    if ax2.get_legend_handles_labels()[0]:  # sin evaluations.npz no hay curvas
        ax2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")

    fig.suptitle(f"Afinamiento con Optuna: {name}", color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def summarize(db: Path, out_dir: Path = DOCS, fig_dir: Path = FIG) -> Path:
    """Genera el .md, el .csv y la figura de un estudio. Devuelve la ruta del .md."""
    name = db.stem
    algo = name.split("_level")[0]
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{db.as_posix()}")
    rows = trial_rows(study, db.parent / name)
    algo_cfg = load_yaml(REPO_ROOT / "configs" / "algos" / f"{algo}.yaml")
    tune_cfg = load_yaml(TUNE_CONFIG)
    env = study.user_attrs.get("protocol", {}).get("env", {})
    max_value = env.get("max_episode_steps")  # recompensa ≤ 1 por paso (§4.4): el máximo es la duración del episodio

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure = f"afinamiento_{name}.png"
    plot_study(name, rows, max_value, fig_dir / figure)
    write_csv(rows, out_dir / f"{name}.csv")
    md = out_dir / f"{name}.md"
    md.write_text(render_markdown(name, study, rows, algo_cfg, tune_cfg, figure), encoding="utf-8")
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume los estudios de afinamiento de Optuna (E10)")
    parser.add_argument("--algo", nargs="*", default=None, help="por defecto, todos los estudios encontrados")
    parser.add_argument("--storage-dir", default=None, help="por defecto storage_dir de tune.yaml")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = REPO_ROOT / (args.storage_dir or load_yaml(TUNE_CONFIG)["storage_dir"])
    dbs = sorted(storage.glob("*_level*.db"))
    if args.algo:
        dbs = [db for db in dbs if db.stem.split("_level")[0] in args.algo]
    if not dbs:
        raise SystemExit(f"No hay estudios en {storage}")
    for db in dbs:
        print(summarize(db))


if __name__ == "__main__":
    main()
