"""Videos de los controladores finales (presentación del informe).

Uso:
    python -m analysis.videos_finales [--runs-dir results/analisis] [--out results/videos_finales]

Por algoritmo se toma la corrida **mediana** del nivel 2 (retorno final = media de las últimas 3 evaluaciones), no
la mejor: es la representativa. Se graban el PID y los 4 agentes, con la misma semilla, en tres escenarios:

- `nivel2`: comandos de velocidad y giro con empujones (condición de entrenamiento).
- `nivel3`: además, ruido de sensores, latencia y física aleatorizada (ningún agente la vio al entrenar).
- `empujon`: nivel 0 con un único empujón hacia adelante a t = 2 s (fuerza en `--push-force`).

Deja un `.mp4` por política y escenario y un mosaico por escenario (`mosaico_<escenario>.mp4`) con los 5
controladores lado a lado. Las políticas se cargan con el mismo código que la evaluación (`evaluation/`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis.finales import ALGO_ORDER, final_return, load_runs
from envs.balance_bot_env import REPO_ROOT, BalanceBotEnv
from evaluation.evaluate import PIDPolicy, SB3Policy, load_eval_config

TILE = (400, 300)
FRAME_STRIDE = 2  # control a 100 Hz; video a 50 fps


def median_runs(runs_dir: Path, level: int = 2) -> dict[str, Path]:
    runs = [r for r in load_runs(runs_dir) if r.level == level]
    out = {}
    for algo in ALGO_ORDER:
        mine = sorted((r for r in runs if r.algo == algo), key=lambda r: final_return(r, 3))
        if mine:
            out[algo] = mine[(len(mine) - 1) // 2].path
    return out


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def episode_frames(level: int, make_policy, seed: int, push_force: float | None) -> tuple[list[np.ndarray], bool, float]:
    env = BalanceBotEnv(level=level, render_mode="rgb_array")
    policy = make_policy(env)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    if push_force is not None:
        env.perturbations.schedule(2.0, push_force, 0.0)
    frames, ret, fell, k = [env.render()], 0.0, False, 0
    while True:
        obs, r, terminated, truncated, _ = env.step(np.asarray(policy.act(obs), dtype=np.float32))
        ret += float(r)
        k += 1
        if k % FRAME_STRIDE == 0 or terminated:
            frames.append(env.render())
        if terminated or truncated:
            fell = terminated
            break
    env.close()
    return frames, fell, ret


def label(frame: np.ndarray, title: str, status: str | None, size: tuple[int, int] = TILE) -> np.ndarray:
    img = Image.fromarray(frame).resize(size)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], 30], fill=(255, 255, 255))
    draw.text((8, 4), title, fill=(11, 11, 11), font=_font(18))
    if status:
        color = (235, 104, 52) if status.startswith("CAYÓ") else (27, 175, 122)  # naranja: cayó; verde: completo
        draw.rectangle([0, size[1] - 30, size[0], size[1]], fill=color)
        draw.text((8, size[1] - 26), status, fill=(255, 255, 255), font=_font(18))
    return np.asarray(img)


def mosaic(tiles: list[list[np.ndarray]], cols: int = 3) -> list[np.ndarray]:
    n_frames = max(len(t) for t in tiles)
    blank = np.full_like(tiles[0][0], 252)
    out = []
    for i in range(n_frames):
        frames = [t[min(i, len(t) - 1)] for t in tiles] + [blank] * (-len(tiles) % cols)
        rows = [np.concatenate(frames[j:j + cols], axis=1) for j in range(0, len(frames), cols)]
        out.append(np.concatenate(rows, axis=0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Videos de los controladores finales")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "results" / "analisis"))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "videos_finales"))
    parser.add_argument("--seed", type=int, default=None, help="por defecto, la primera semilla de la evaluación final")
    parser.add_argument("--push-force", type=float, default=10.0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(load_eval_config()["seed_offset"])
    runs = median_runs(Path(args.runs_dir))
    policies: list[tuple[str, object]] = [("PID", PIDPolicy)]
    for algo, path in runs.items():
        sb3 = SB3Policy(path)
        policies.append((f"{algo.upper()} (semilla {path.name.rsplit('seed', 1)[1]})", lambda env, p=sb3: p))
    print("Corridas medianas:", {a: p.name for a, p in runs.items()})

    scenarios = {"nivel2": (2, None), "nivel3": (3, None), "empujon": (0, args.push_force)}
    fps = int(round(100 / FRAME_STRIDE))
    for name, (level, force) in scenarios.items():
        tiles = []
        for title, make in policies:
            frames, fell, ret = episode_frames(level, make, seed, force)
            tag = title.split(" ")[0].lower()
            imageio.mimsave(out / f"{tag}_{name}.mp4", [label(f, title, None, f.shape[1::-1]) for f in frames], fps=fps, macro_block_size=1)
            n = len(frames)
            tiles.append([label(f, title, (f"CAYÓ · retorno {ret:.0f}" if fell else f"retorno {ret:.0f}") if i == n - 1 else None) for i, f in enumerate(frames)])
            print(f"{name}: {title}: {'cayó' if fell else 'completo'}, retorno {ret:.0f}")
        imageio.mimsave(out / f"mosaico_{name}.mp4", mosaic(tiles), fps=fps, macro_block_size=1)
        print(out / f"mosaico_{name}.mp4")


if __name__ == "__main__":
    main()
