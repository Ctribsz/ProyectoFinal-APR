# Fase 1 — Entorno MuJoCo y controlador PID

> **Estado: completa (2026-11-02).** Los cuatro criterios de salida se cumplen. Resultados en `docs/entregables.md` (E1–E4).

> Plan de trabajo de la fase activa. Referencia: secciones 4 y 5 de `especificacion-proyecto.md`.

## Objetivo

Tener un robot simulado que se pueda controlar desde un entorno de `gymnasium`, y un controlador PID que lo mantenga en equilibrio. Al cerrar esta fase sabemos que el modelo físico es correcto y que el problema se puede resolver, antes de gastar tiempo entrenando agentes.

## Criterios de salida

La fase termina cuando se cumplen los cuatro:

1. `check_env(BalanceBotEnv(level=0))` pasa sin errores ni advertencias.
2. El PID mantiene el robot en pie durante los 10 s del episodio en el nivel 0, en 20 de 20 episodios con semillas distintas.
3. El render muestra el robot con proporciones correctas y movimiento físicamente creíble: sin piezas que se atraviesan, sin rebotes raros ni ruedas que patinan sin razón.
4. `pytest` pasa completo.

## Decisiones que afectan esta fase

| Decisión | Cómo se maneja en F1 |
|---|---|
| D1 — Torque ideal vs. motor DC | Se empieza con torque ideal, pero la conversión acción → torque vive en una sola función (`_apply_action`) para poder cambiarla después sin tocar el resto. |
| D2 — PID vs. LQR | Se empieza con PID en cascada porque no requiere linealizar. El LQR queda como opción si el PID cuesta sintonizarlo. |

Los niveles 1–3 (perturbaciones, comandos, aleatorización) **no** son parte de esta fase. El constructor acepta `level`, pero en F1 solo tiene que funcionar el nivel 0. Dejar los niveles superiores lanzando `NotImplementedError` es aceptable.

---

## Tareas

| # | Tarea | Responsable | Depende de |
|---|---|---|---|
| T1 | Modelo MJCF del robot | Christian | — |
| T2 | Esqueleto de `BalanceBotEnv` | Christian | — |
| T3 | Observación, acción, recompensa y terminación | Christian | T1, T2 |
| T4 | Render | Christian | T2 |
| T5 | Controlador PID | Christian | T3 |
| T6 | Pruebas | Christian | T3 |
| T7 | Configuración YAML del nivel 0 | Christian | T3 |

T1 y T2 se pueden hacer en paralelo desde el primer día.

En paralelo a esta fase, Mario prepara las configuraciones de DDPG y PPO (entregable E5 en `entregables.md`), que no dependen del entorno.

---

### T1 — Modelo MJCF (`assets/balance_bot.xml`)

**Estructura de cuerpos:**
- `chassis`: cuerpo principal con `freejoint`. Una caja o cilindro alto con la masa de la especificación 4.1, centro de masa por encima del eje de las ruedas.
- `wheel_left` y `wheel_right`: hijos de `chassis`, cada uno con un `hinge` cuyo eje es el eje de las ruedas, y geometría `cylinder`.
- Un `site` llamado `imu` en el chasis, a la altura donde iría la IMU.
- Un plano como suelo.

**Actuadores:**
- Un `motor` por rueda con `ctrlrange="-1 1"` y `gear` igual al torque máximo (0.3 N·m según la especificación). Así el control en [-1, 1] queda escalado al torque directamente en el MJCF y no hay constantes físicas en el código.

**Sensores** (se leen igual que en un robot real):
- `framequat` en el site `imu`, para obtener la orientación.
- `gyro` en el site `imu`, para las velocidades angulares.
- `jointvel` en cada rueda, como encoder.
- Opcional: `accelerometer` en el site `imu`, por si después se quiere estimar el ángulo como lo haría un robot real.

**Física:**
- `timestep="0.002"`, siguiendo la especificación.
- Integrador `implicitfast` como punto de partida. Si hay inestabilidad, probar `RK4`.
- Evitar colisiones entre chasis y ruedas (con `contype`/`conaffinity` o con `<exclude>`). Solo las ruedas tocan el suelo.
- Ajustar la fricción rueda-suelo para que no patinen con el torque máximo.

**Cómo verificar:** abrir el modelo con `python -m mujoco.viewer --mjcf=assets/balance_bot.xml`. Sin control, el robot debe caerse hacia un lado de forma natural, sin explotar ni atravesar el suelo.

---

### T2 — Esqueleto de `BalanceBotEnv` (`envs/balance_bot_env.py`)

Subclase directa de `gymnasium.Env` usando los bindings de `mujoco`. Se recomienda no heredar de `gymnasium.envs.mujoco.MujocoEnv`, porque su API cambia entre versiones y se necesita control fino sobre los sensores.

**Interfaz:**
```python
class BalanceBotEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, level: int = 0, config: dict | None = None,
                 render_mode: str | None = None): ...
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]: ...
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]: ...
    def render(self): ...
    def close(self): ...
```

**Puntos de cuidado:**
- `reset` debe llamar a `super().reset(seed=seed)` y usar `self.np_random` para todo lo aleatorio. Sin esto, las semillas no son reproducibles.
- `step` aplica la acción y avanza `frame_skip` (5) pasos de `mj_step`, para quedar en control a 100 Hz.
- La acción se recorta a [-1, 1] antes de aplicarla.

---

### T3 — Observación, acción, recompensa y terminación

Seguir exactamente las secciones 4.2–4.5 de la especificación.

**Observación (9 valores):** pitch, velocidad de pitch, velocidad de yaw, velocidad de cada rueda, comandos v* y ψ̇* (en cero en el nivel 0) y la acción anterior.

**Cuidado con el pitch:** se obtiene del cuaternión del sensor `framequat`. Hay que fijar la convención de ejes (cuál es adelante, cuál es el eje de las ruedas) y verificar el signo: inclinar el robot hacia adelante a mano en el viewer y confirmar que el pitch observado crece en el sentido esperado. Un signo invertido aquí arruina el PID y el aprendizaje sin dar ningún error.

**Estado inicial:** pitch uniforme en [-0.05, 0.05] rad, aplicado rotando el cuaternión del `freejoint`. Velocidades en cero. Llamar a `mj_forward` después de modificar el estado.

**Recompensa:** cada componente de la especificación 4.4 se calcula por separado y se devuelve en `info["reward_components"]` como diccionario. La recompensa total es la suma ponderada, con pesos leídos de la configuración.

**Terminación:** `terminated=True` si `|pitch| > 0.6`. `truncated=True` al llegar a 1000 pasos.

---

### T4 — Render

- `rgb_array`: usar `mujoco.Renderer`, con una cámara que siga al chasis.
- `human`: usar `mujoco.viewer.launch_passive` y sincronizar en cada `step`.
- Agregar una función para grabar un episodio a `.mp4`, porque va a servir para el README y la presentación.

---

### T5 — Controlador PID (`baselines/pid_controller.py`)

**PID en cascada:**
- **Lazo externo (velocidad):** recibe el error entre la velocidad comandada y la velocidad medida (promedio de las ruedas × radio) y produce un pitch objetivo. Para el nivel 0, la velocidad comandada es cero. Sin este lazo, el robot se mantiene vertical pero se va desplazando sin control.
- **Lazo interno (ángulo):** recibe el error entre el pitch objetivo y el pitch medido y produce el torque.
- Para el giro (nivel 2 en adelante) se suma un término diferencial entre ruedas. No es necesario en F1.

**Reglas:**
- El PID usa **solo la observación** del entorno, igual que los agentes. No puede leer el estado interno del simulador. Así la comparación es justa.
- Expone la misma interfaz que una política de SB3, `predict(obs) -> action`, para que `evaluate.py` lo trate igual que a cualquier agente.
- Las ganancias van en `configs/baselines/pid.yaml`.

**Sintonía sugerida:** primero solo el lazo interno con P, subiendo hasta que oscile, luego agregar D para amortiguar. Después se agrega el lazo externo con ganancias pequeñas.

---

### T6 — Pruebas (`tests/test_env.py`)

| Prueba | Qué verifica |
|---|---|
| `test_check_env` | `check_env` de gymnasium pasa |
| `test_determinism` | Misma semilla y mismas acciones dan observaciones idénticas |
| `test_obs_valid` | La observación tiene la forma correcta y no contiene NaN ni infinitos |
| `test_falls_without_control` | Con acción cero, el episodio termina por caída antes de los 1000 pasos |
| `test_termination_threshold` | `terminated` se activa justo al pasar `|pitch| > 0.6` |
| `test_reward_components` | La suma ponderada de `info["reward_components"]` coincide con la recompensa devuelta |
| `test_pitch_sign` | Aplicar torque positivo en ambas ruedas inclina el robot en la dirección documentada |
| `test_pid_balances` | El PID sobrevive 1000 pasos en 20 semillas del nivel 0 |

---

### T7 — Configuración del nivel 0 (`configs/env/level_0.yaml`)

```yaml
level: 0
frame_skip: 5
max_episode_steps: 1000
termination_pitch: 0.6
initial_pitch_range: 0.05

reward_weights:
  alive: 1.0
  theta: 1.0
  dtheta: 0.1
  vel: 0.0
  yaw: 0.0
  effort: 0.01
  smooth: 0.05

perturbations:
  enabled: false

commands:
  enabled: false

randomization:
  enabled: false
```

---

## Al cerrar la fase

- Actualizar la tabla de estado en `README.md`.
- Anotar en la bitácora de la especificación cualquier valor que haya cambiado (dimensiones, fricción, torque, convención de ejes).
- Grabar un video corto del PID balanceando.
- Escribir `docs/fase-2-prueba-de-humo.md`.
