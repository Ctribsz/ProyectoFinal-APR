"""Pruebas del render y la grabación (entregable E3). Se omiten si no hay contexto OpenGL."""

import warnings

import imageio.v2 as imageio
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from envs import BalanceBotEnv
from envs.recording import ZeroPolicy, record_episode


@pytest.fixture
def env() -> BalanceBotEnv:
    e = BalanceBotEnv(level=0, render_mode="rgb_array")
    try:
        e.reset(seed=0)
        e.render()
    except Exception as exc:  # sin pantalla ni EGL/OSMesa
        e.close()
        pytest.skip(f"Render no disponible: {exc}")
    yield e
    e.close()


def test_render_rgb(env: BalanceBotEnv) -> None:
    frame = env.render()
    cfg = env.config["render"]
    assert frame.shape == (cfg["height"], cfg["width"], 3)
    assert frame.dtype == np.uint8
    assert frame.std() > 5  # no es una imagen vacía


def test_check_env_with_render(env: BalanceBotEnv) -> None:
    # check_env pide crear el entorno con gym.make para probar los otros modos de render. Eso abriría el
    # viewer "human", que no se prueba aquí (falla con GLFW en Wayland), así que solo se ignora ese aviso.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message=".*Not able to test alternative render modes.*")
        check_env(env)


def test_camera_follows_robot(env: BalanceBotEnv) -> None:
    """La cámara sigue al chasis: tras avanzar, el robot sigue en el cuadro (el cuadro cambia poco)."""
    first = env.render().astype(float)
    for _ in range(30):
        env.step(np.array([0.2, 0.2], dtype=np.float32))
    assert env.data.qpos[0] > 0.01
    assert np.abs(env.render().astype(float) - first).mean() < 40


def test_record_episode(env: BalanceBotEnv, tmp_path) -> None:
    out = tmp_path / "caida.mp4"
    summary = record_episode(env, ZeroPolicy(), out, seed=0)
    assert out.exists() and out.stat().st_size > 0
    assert summary["terminated"]
    reader = imageio.get_reader(out)
    n_frames = sum(1 for _ in reader)
    reader.close()
    assert n_frames == summary["steps"] + 1
