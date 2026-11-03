"""Pruebas del modelo MJCF (entregable E1, especificación §4.1)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "assets" / "balance_bot.xml"


@pytest.fixture
def model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


def _pitch(data: mujoco.MjData) -> float:
    """Pitch (rotación alrededor de +y) a partir del cuaternión de la IMU."""
    w, x, y, z = data.sensor("imu_quat").data
    return float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))


def _set_pitch(data: mujoco.MjData, theta: float) -> None:
    data.qpos[3:7] = [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0]


def _settle(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 100) -> None:
    """Deja que los contactos se asienten con el chasis fijo en vertical."""
    for _ in range(steps):
        mujoco.mj_step(model, data)
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        data.qvel[:] = 0.0


def test_model_structure(model: mujoco.MjModel) -> None:
    assert model.opt.timestep == pytest.approx(0.002)
    assert model.nu == 2
    for name in ("motor_left", "motor_right"):
        act = model.actuator(name)
        assert act.gear[0] == pytest.approx(0.3)
        np.testing.assert_allclose(act.ctrlrange, [-1.0, 1.0])
    for name in ("imu_quat", "imu_gyro", "imu_accel", "enc_left", "enc_right"):
        model.sensor(name)  # lanza KeyError si no existe


def test_physical_parameters(model: mujoco.MjModel) -> None:
    chassis = model.body("chassis").id
    assert model.body_subtreemass[chassis] == pytest.approx(1.1)
    assert model.geom("wheel_left").size[0] == pytest.approx(0.04)
    separation = model.body("wheel_left").pos[1] - model.body("wheel_right").pos[1]
    assert separation == pytest.approx(0.18)


def test_only_wheels_touch_ground_upright(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    _settle(model, data)
    pairs = {
        frozenset((model.geom(data.contact[k].geom1).name, model.geom(data.contact[k].geom2).name))
        for k in range(data.ncon)
    }
    assert pairs == {frozenset(("floor", "wheel_left")), frozenset(("floor", "wheel_right"))}


def test_falls_without_control(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    _set_pitch(data, 0.05)
    mujoco.mj_forward(model, data)
    fell = False
    for _ in range(1500):  # 3 s
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        fell |= abs(_pitch(data)) > 0.6
    assert fell
    assert _pitch(data) > 0  # cae hacia adelante, del lado de la inclinación inicial
    assert all(data.contact[k].dist > -2e-3 for k in range(data.ncon))  # sin atravesar el suelo


def test_pitch_sign(model: mujoco.MjModel) -> None:
    """Torque > 0 en ambas ruedas: avanza hacia +x y el chasis se inclina hacia atrás (θ < 0)."""
    data = mujoco.MjData(model)
    _settle(model, data)
    data.ctrl[:] = 0.3
    for _ in range(25):
        mujoco.mj_step(model, data)
    assert data.qvel[0] > 0
    assert _pitch(data) < 0
    assert data.sensor("enc_left").data[0] > 0


def test_yaw_sign(model: mujoco.MjModel) -> None:
    """ψ̇ > 0 (antihorario visto desde arriba) cuando la rueda derecha va más rápido que la izquierda."""
    data = mujoco.MjData(model)
    _settle(model, data)
    data.ctrl[:] = [-0.2, 0.2]
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert data.sensor("imu_gyro").data[2] > 0


def test_no_slip_moderate_torque(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    _settle(model, data)
    data.ctrl[:] = 0.3
    radius = model.geom("wheel_left").size[0]
    for _ in range(50):  # 0.1 s
        mujoco.mj_step(model, data)
        wheel_abs = data.sensor("enc_left").data[0] + data.qvel[4]  # rueda relativa + pitch del chasis
        assert abs(wheel_abs * radius - data.qvel[0]) < 0.05
