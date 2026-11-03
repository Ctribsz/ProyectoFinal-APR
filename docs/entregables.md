# Entregables y cronograma

> El trabajo se organiza en entregables. Cada uno tiene responsable, archivos, criterio de aceptación y dependencias.
> Referencia: `especificacion-proyecto.md` (§10 reparto, §13 D7 fechas).

## Fechas clave

| Fecha | Hito |
|---|---|
| 2026-11-02 | Avance del proyecto (E0) |
| ~2026-11-13 | Entrega final |

## Reparto

- **Christian:** entorno, PID, pipeline común de entrenamiento, afinamiento y diagnósticos, evaluación, y los algoritmos TD3 y SAC.
- **Mario:** DDPG y PPO con SB3: configuración, espacio de búsqueda, afinamiento, corridas finales y análisis de sus diagnósticos.
- **Ambos:** análisis estadístico, informe y dictamen.

**Regla:** los cuatro algoritmos pasan por el mismo `train.py`, `tune.py` y `callbacks.py`, y solo cambia su YAML en `configs/algos/`. Si un algoritmo necesita algo que el pipeline no soporta, se pide el cambio en el pipeline. No se crea un script paralelo.

## Entregables

| # | Entregable | Responsable | Fase | Depende de | Fecha objetivo | Estado |
|---|---|---|---|---|---|---|
| E0 | Avance del 2 de noviembre | Ambos | — | — | 2026-11-02 | En progreso |
| E1 | Modelo MJCF (`assets/balance_bot.xml`) | Christian | F1 | — | 2026-11-02 | Listo |
| E2 | `BalanceBotEnv` nivel 0 + `configs/env/level_0.yaml` + pruebas | Christian | F1 | E1 | 2026-11-03 | Listo |
| E3 | Render (`rgb_array`, `human`) y grabación a `.mp4` | Christian | F1 | E2 | 2026-11-03 | Listo (ver nota) |
| E4 | PID en cascada + `configs/baselines/pid.yaml` | Christian | F1 | E2 | 2026-11-04 | Listo |
| E5 | `configs/algos/ddpg.yaml` y `ppo.yaml`: valores por defecto y espacio de búsqueda de Optuna | Mario | F1 (en paralelo) | — | 2026-11-04 | Pendiente |
| E6 | `training/train.py` común (los 4 algoritmos, semilla, YAML y versiones guardados) | Christian | F2 | E2 | 2026-11-05 | Listo |
| E7 | Prueba de humo en el nivel 0: TD3/SAC (Christian), DDPG/PPO (Mario) | Ambos | F2 | E5, E6 | 2026-11-05 | TD3/SAC hechos; DDPG/PPO pendientes |
| E8 | Niveles 1–3: perturbaciones, comandos, aleatorización | Christian | F2→F3 | E2 | 2026-11-06 | Listo |
| E9 | `training/tune.py` + `training/callbacks.py` (diagnósticos §8) | Christian | F3 | E6 | 2026-11-06 | Pendiente |
| E10 | Afinamiento con Optuna: TD3/SAC (Christian), DDPG/PPO (Mario) | Ambos | F3 | E8, E9 | 2026-11-08 | Pendiente |
| E11 | `evaluation/evaluate.py` + `evaluation/push_test.py` | Christian | F4 | E4, E8 | 2026-11-08 | Pendiente |
| E12 | Corridas finales con semillas y E1–E6 de §7.3: cada quien sus dos algoritmos | Ambos | F4 | E10, E11 | 2026-11-10 | Pendiente |
| E13 | Análisis con `rliable`, hipótesis H1–H4, informe y dictamen | Ambos | F5 | E12 | 2026-11-13 | Pendiente |

## Criterios de aceptación

- **E1:** el modelo abre en `python -m mujoco.viewer`; sin control el robot cae de forma natural, sin explotar ni atravesar el suelo.
- **E2:** `check_env` pasa sin advertencias; pasan las pruebas T6 de `fase-1-entorno.md` (salvo `test_pid_balances`).
- **E3:** video de un episodio guardado; proporciones correctas.
- **E4:** el PID sobrevive 1000 pasos en 20 de 20 semillas del nivel 0 (`test_pid_balances`).
- **E5:** los YAML cargan en `train.py`; los rangos de búsqueda cubren los hiperparámetros de §6.
- **E6:** una corrida corta de cada algoritmo termina y deja en `results/<id>/` la semilla, el YAML y las versiones de las librerías.
- **E7:** los 4 algoritmos mejoran su retorno en el nivel 0 (criterio de F2).
- **E8:** cada nivel tiene su YAML y sus pruebas; perturbaciones y aleatorización usan `self.np_random`.
- **E9:** el estudio de Optuna se guarda en `.db`; los callbacks registran la sobreestimación de Q, α, KL/clip, la suavidad y los componentes de la recompensa.
- **E10:** las mejores configuraciones quedan guardadas en `configs/algos/`.
- **E11:** el PID y un agente de SB3 se evalúan con el mismo código.
- **E12:** todas las semillas completas para los 4 algoritmos.
- **E13:** H1–H4 contrastadas con IQM e intervalos bootstrap.

## Notas de entregables cerrados

- **E3:** el modo `human` usa `mujoco.viewer.launch_passive`, que da segfault con GLFW en Wayland (la ventana se dibuja desde un hilo secundario). En esa configuración se usa la grabación a `.mp4` (`python -m envs.recording`) o el viewer bloqueante `python -m mujoco.viewer --mjcf=assets/balance_bot.xml`.

- **E4:** ganancias elegidas por búsqueda en rejilla sobre 20 semillas del nivel 0. El lazo interno es estable en un rango amplio (kp 4–20, kd 0.2–0.8) y aguanta θ0 de ±0.25 rad. El lazo externo debe ser pequeño: con kp_v ≥ 0.2 se vuelve inestable, porque para frenar el robot primero acelera las ruedas (fase no mínima). Resultado: 20/20 episodios completos, retorno medio 999.5, deriva final de unos 1 mm. Video en `results/videos/pid_balanceo.mp4`.
- **E6:** `train.py` común, con `configs/training/common.yaml` y `configs/algos/{td3,sac}.yaml`. Cada corrida guarda `config.yaml`, `meta.json` (semilla, versiones, commit, máquina, duración), `evaluations.npz`, `model.zip` y TensorBoard con cada componente de la recompensa. Hay 10 pruebas que corren los 4 algoritmos.
- **E8:** niveles 1–3 en `envs/perturbations.py`, `envs/commands.py` y `envs/randomization.py`, con 24 pruebas. Resultados del PID (20 semillas): nivel 1 sin caídas, nivel 2 sin caídas con kp_yaw = 0.2 (RMSE de ψ̇ 0.13), **nivel 3: 13/20 caídas**. Con kp = 15, el ruido de θ (0.01 rad) se traduce en ±0.15 de acción, y hay hasta 20 ms de latencia. Es el escenario donde H4 espera que el RL supere al PID.
- **Cierre de F1 (2026-11-02):** `check_env` sin advertencias (creado con `gym.make`), PID 20/20, render correcto y 29 pruebas pasando.

## Riesgo de cómputo

Con fecha final cercana al 13 de noviembre, los presupuestos propuestos en la especificación (40 pruebas × 200k pasos por algoritmo en §6, y 10 semillas × 1M pasos en §7.1) pueden no caber. La decisión D4 sigue abierta: se fija con los tiempos medidos en E7, antes de lanzar E10.

### Mediciones del 2026-11-01 (nivel 0, MLP [256, 256], CPU, 1 hilo de torch por proceso)

| Algoritmo | Pasos/s (1 proceso) | 1M pasos |
|---|---|---|
| SAC | 107 | ~2.6 h |
| TD3 | 171 | ~1.6 h |
| DDPG | 200 | ~1.4 h |
| PPO | 1946 | ~9 min |

En paralelo, la máquina de Christian (16 hilos) rinde como unos 7 procesos independientes: 14 procesos de SAC suman 745 pasos/s.

Estimación con los presupuestos de la especificación, todo en esa máquina:

| Bloque | Horas de CPU | Tiempo real (~7× en paralelo) |
|---|---|---|
| E10: afinamiento (40 × 200k × 4 algoritmos, sin contar la poda) | ~46 h | ~6.5 h |
| E12: corridas finales (10 semillas × 1M × 4 algoritmos) | ~58 h | ~8 h |
| E12: experimento E5, entrenar en el nivel 3 (10 semillas × 1M × 4 algoritmos) | ~58 h | ~8 h |
| **Total** | **~160 h** | **~23 h** |

Cabe en el cronograma si las corridas se lanzan en bloque (por ejemplo, de noche). Que Mario corra DDPG y PPO en su propia máquina reduce el tiempo total.
