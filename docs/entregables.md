# Entregables y cronograma

> El trabajo se organiza en entregables. Cada uno tiene responsable, archivos, criterio de aceptación y dependencias.
> Referencia: `especificacion-proyecto.md` (§10 reparto, §13 D7 fechas).

## Fechas clave

| Fecha | Hito |
|---|---|
| 2026-11-02 | Avance del proyecto (E0) |
| 2026-11-09 | Entrega final (lunes, medianoche). Antes se planeaba alrededor del 13 de noviembre |

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
| E5 | `configs/algos/ddpg.yaml` y `ppo.yaml`: valores por defecto y espacio de búsqueda de Optuna | Mario | F1 (en paralelo) | — | 2026-11-04 | Listo |
| E6 | `training/train.py` común (los 4 algoritmos, semilla, YAML y versiones guardados) | Christian | F2 | E2 | 2026-11-05 | Listo |
| E7 | Prueba de humo en el nivel 0: TD3/SAC (Christian), DDPG/PPO (Mario) | Ambos | F2 | E5, E6 | 2026-11-05 | Listo (ver nota) |
| E8 | Niveles 1–3: perturbaciones, comandos, aleatorización | Christian | F2→F3 | E2 | 2026-11-06 | Listo |
| E9 | `training/tune.py` + `training/callbacks.py` (diagnósticos §8) | Mario | F3 | E6 | 2026-11-06 | Listo |
| E10 | Afinamiento con Optuna: TD3/SAC (Christian), DDPG/PPO (Mario) | Ambos | F3 | E8, E9 | 2026-11-08 | DDPG y PPO listos; TD3/SAC: Christian |
| E11 | `evaluation/evaluate.py` + `evaluation/push_test.py` | Christian | F4 | E4, E8 | 2026-11-08 | Pendiente |
| E12 | Corridas finales con semillas y E1–E6 de §7.3: cada quien sus dos algoritmos | Ambos | F4 | E10, E11 | 2026-11-09 | En progreso (ver `reparto-e12.md`) |
| E13 | Análisis con `rliable`, hipótesis H1–H4, informe y dictamen | Ambos | F5 | E12 | 2026-11-09 | En progreso: `analysis/finales.py` listo |

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

- **E5:** valores por defecto de SB3 2.x. DDPG usa ruido gaussiano con sigma 0.1, igual que TD3, porque SB3 no trae ruido por defecto. Los rangos de búsqueda de los hiperparámetros compartidos con TD3/SAC son los mismos. PPO queda con `n_envs: 1`; usar más entornos en paralelo es una decisión pendiente antes de F3.

- **E7:** prueba de humo de DDPG/PPO en el nivel 0 con las semillas 0 y 1 (figura en `docs/img/prueba_humo_ddpg_ppo.png`). PPO llega a 1000 desde los 20k pasos en ambas semillas, con caídas puntuales de las que se recupera. DDPG aprende (mejor retorno: 854 y 715), pero se desestabiliza y termina en 243 y 70. Cumple el criterio, aunque con poco margen en la semilla 1. Los pasos/s de la laptop de Mario (i7-1065G7, 4 corridas en paralelo: DDPG ~31, PPO ~112) no sirven para D4; se midieron de nuevo con la máquina libre el 2026-11-06 (ver «Riesgo de cómputo»). En Windows, `train.py` termina con código de salida 1 por la flecha `→` del resumen final (consola `cp1252`), aunque la corrida se guarde completa.
- **E6:** `train.py` común, con `configs/training/common.yaml` y `configs/algos/{td3,sac}.yaml`. Cada corrida guarda `config.yaml`, `meta.json` (semilla, versiones, commit, máquina, duración), `evaluations.npz`, `model.zip` y TensorBoard con cada componente de la recompensa. Hay 10 pruebas que corren los 4 algoritmos.
- **E8:** niveles 1–3 en `envs/perturbations.py`, `envs/commands.py` y `envs/randomization.py`, con 24 pruebas. Resultados del PID (20 semillas): nivel 1 sin caídas, nivel 2 sin caídas con kp_yaw = 0.2 (RMSE de ψ̇ 0.13), **nivel 3: 13/20 caídas**. Con kp = 15, el ruido de θ (0.01 rad) se traduce en ±0.15 de acción, y hay hasta 20 ms de latencia. Es el escenario donde H4 espera que el RL supere al PID.
- **E9 (2026-11-06, Mario):** `training/tune.py` (Optuna con TPE y podador por mediana; el estudio en `results/optuna/<algo>_level<n>.db`; varios procesos con `lab/afinar.py`, que se reanuda solo) y `training/callbacks.py` con `DiagnosticsCallback`: sobreestimación de Q (método del artículo de TD3, ver §8) y suavidad de la acción, parametrizados en `configs/training/diagnostics.yaml`. Solo los usan las corridas finales. α de SAC y KL/clip de PPO los registra SB3 en TensorBoard; hay pruebas que lo verifican. `train()` anota el estado de git al inicio de la corrida.
- **E10, DDPG y PPO (2026-11-07, Mario):** 40 pruebas × 200k pasos en el nivel 2 cada uno, en la laptop con 4 procesos. DDPG: 7.6 h, mejor prueba #39 (objetivo 836.3: lr 8.852e-4, σ 0.2357, τ 0.01708, batch 256). PPO: 2.6 h, mejor prueba #34 (objetivo 865.9: lr 1.607e-4, clip 0.1628, n_epochs 16, n_steps 2048, λ 0.9498). Detalle y figuras en `docs/afinamiento/` (`python -m analysis.resumen_afinamiento`). Limitación: una semilla por prueba.
- **Cierre de F1 (2026-11-02):** `check_env` sin advertencias (creado con `gym.make`), PID 20/20, render correcto y 29 pruebas pasando.

## Riesgo de cómputo

Con fecha final cercana al 13 de noviembre, los presupuestos propuestos en la especificación (40 pruebas × 200k pasos por algoritmo en §6, y 10 semillas × 1M pasos en §7.1) pueden no caber. La decisión D4 sigue abierta: se fija con los tiempos medidos en E7, antes de lanzar E10.

**Estado al 2026-11-08:** E10 se corrió con el presupuesto de §6 (40 × 200k) para DDPG y PPO. E12 se reparte entre la laptop de Mario, Colab y otras PC (`docs/reparto-e12.md`). Propuesta para D4, pendiente de confirmar con Christian: 10 semillas en el nivel 2 y 5 en el nivel 3 (E5), iguales para los 4 algoritmos.

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

### Mediciones del 2026-11-06 (laptop de Mario: i7-1065G7, 4 núcleos; 1 proceso, 1 hilo de torch, con cargador)

Con `train()` y los YAML del repo; incluyen el costo de las evaluaciones (10 episodios cada 10k pasos).

| Algoritmo | Pasos/s, nivel 0 | Pasos/s, nivel 2 |
|---|---|---|
| DDPG | 52 | 58 |
| PPO | 253 | 410 |

- La laptop rinde unas 3.5× menos que la máquina de Christian. PPO es más lento en el nivel 0 porque ya balancea y sus evaluaciones duran los 1000 pasos completos; en el nivel 2 los empujones acortan los episodios.
- Las cifras pueden estar algo bajas: con la máquina en reposo el CPU estaba al 39–46 % por otros procesos. El paralelismo no se midió; por E7 (4 corridas juntas), rinde unas 2×.

Estimación para DDPG y PPO en esta laptop (la poda supone las 5 primeras pruebas completas y la mitad del resto podada hacia los 75k pasos):

| Bloque | DDPG | PPO |
|---|---|---|
| E10: 40 pruebas × 200k en el nivel 2, con poda y ~2× en paralelo | ~14 h | ~2.5 h |
| E12: 10 semillas × 1M (cada bloque), ~2× en paralelo | ~24 h | ~3.5 h |

E10 cabe en la laptop. E12 de DDPG no: los dos bloques (corridas finales y experimento E5) suman unas 48 h reales entre el 2026-11-08 y el 2026-11-10. Opciones para D4: correr DDPG en la máquina de Christian, bajar el presupuesto de E12 para los 4 algoritmos por igual (§6), o ambas.
