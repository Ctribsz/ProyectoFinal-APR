# Fase 2 — Prueba de humo con los 4 algoritmos

> Plan de trabajo de la fase activa. Referencia: secciones 6, 7 y 10 de `especificacion-proyecto.md`; entregables E5–E8 de `entregables.md`.

## Objetivo

Confirmar que el entorno se puede aprender con los 4 algoritmos usando sus valores por defecto, dejar listo el pipeline común de entrenamiento y medir tiempos reales para fijar el presupuesto de pasos (D4) antes de afinar.

## Criterios de salida

1. `python -m training.train` entrena cualquiera de los 4 algoritmos con solo cambiar `--algo` y guarda en `results/<id>/` la semilla, la configuración YAML completa y las versiones de las librerías.
2. Los 4 algoritmos con valores por defecto **mejoran su retorno de evaluación** en el nivel 0: el último retorno supera claramente al primero (criterio de F2 en §10).
3. Quedan medidos los pasos/s y el tiempo real por algoritmo, y D4 se discute con esos datos.
4. Los niveles 1–3 existen (E8), porque el afinamiento de F3 se hace en el nivel 2.
5. `pytest` pasa completo.

## Tareas

| # | Tarea | Responsable | Entregable | Depende de |
|---|---|---|---|---|
| T1 | Formato de `configs/algos/*.yaml` (este documento) | Christian | E6 | — |
| T2 | `configs/algos/td3.yaml`, `sac.yaml` | Christian | E6 | T1 |
| T3 | `configs/algos/ddpg.yaml`, `ppo.yaml` | Mario | E5 | T1 |
| T4 | `training/train.py` común | Christian | E6 | T1 |
| T5 | Prueba de humo TD3/SAC | Christian | E7 | T2, T4 |
| T6 | Prueba de humo DDPG/PPO | Mario | E7 | T3, T4 |
| T7 | Medición de tiempos y propuesta para D4 | Ambos | E7 | T5, T6 |
| T8 | Niveles 1–3: perturbaciones, comandos, aleatorización | Christian | E8 | — |

T3 y T8 se pueden empezar ya: no dependen de `train.py`.

---

### T1 — Formato de `configs/algos/<algo>.yaml`

Mario necesita este contrato para empezar E5. Un archivo por algoritmo:

```yaml
algo: ppo                  # ddpg | td3 | sac | ppo

# Argumentos que se pasan tal cual al constructor de SB3 (PPO(...), SAC(...), ...).
# En F2 se dejan los valores por defecto de SB3; en F3 se reemplazan por los mejores de Optuna.
hyperparams:
  learning_rate: 3.0e-4
  n_steps: 2048
  # ...

n_envs: 1                  # entornos en paralelo (PPO suele usar más de 1)

# Espacio de búsqueda de Optuna para F3 (§6). Tipos: float (con log opcional), int, categorical.
search_space:
  learning_rate: {type: float, low: 1.0e-5, high: 1.0e-3, log: true}
  clip_range:    {type: float, low: 0.1, high: 0.4}
  n_epochs:      {type: int, low: 3, high: 20}
  n_steps:       {type: categorical, choices: [512, 1024, 2048, 4096]}
  gae_lambda:    {type: float, low: 0.8, high: 0.99}
```

Reglas:
- **La arquitectura de red no va aquí.** Es fija para todos, MLP [256, 256] (§6), y vive en `configs/training/common.yaml` junto con la frecuencia y el número de episodios de evaluación.
- El ruido de exploración de DDPG y TD3 se escribe como `action_noise: {type: normal, sigma: 0.1}`, y `train.py` lo convierte al objeto de SB3.
- Los hiperparámetros de `search_space` son los de la tabla de §6. Agregar otros se discute antes, porque cambia el presupuesto de afinamiento.

### T4 — `training/train.py`

```bash
python -m training.train --algo sac --env-config configs/env/level_0.yaml --seed 0 [--timesteps 100000]
```

- Lee `configs/algos/<algo>.yaml` y `configs/training/common.yaml`; nada de hiperparámetros en el código.
- Crea `results/<algo>_level<n>_seed<s>_<fecha>/` con:
  - `config.yaml`: entorno, algoritmo y común, ya combinados, exactamente como se usaron;
  - `meta.json`: semilla, versiones de las librerías, commit de git, máquina, hilos de torch y duración;
  - logs de TensorBoard, `evaluations.npz` del `EvalCallback` y `model.zip`.
- Evaluación periódica determinista con semillas de evaluación fijas y distintas a las de entrenamiento (§7.2).
- `torch.set_num_threads(1)` por defecto, para correr muchos procesos en paralelo (ver la medición en `entregables.md`).

### T5 / T6 — Prueba de humo

- Nivel 0, valores por defecto de SB3, presupuesto corto: 100k pasos para DDPG, TD3 y SAC, y 300k para PPO.
- 2 semillas por algoritmo.
- Tiempo estimado en un proceso: SAC ~16 min, TD3 ~10 min, DDPG ~8 min, PPO ~3 min.
- Se registra la curva de evaluación, el tiempo real y los pasos/s.

### T7 — Propuesta para D4

Con las curvas de la prueba de humo se estima cuántos pasos hacen falta para converger en el nivel 0 y se recalcula la tabla de cómputo de `entregables.md`. La decisión la toma el equipo.

### T8 — Niveles 1–3

Siguiendo §4.6:
- `envs/perturbations.py`: impulsos con `xfrc_applied` durante 0.1 s, uno cada 2 s en promedio, con magnitud ~ U(0, F_max).
- Comandos v* y ψ̇* que cambian cada 2–4 s, dentro de `v_max` y `yaw_rate_max`.
- `envs/randomization.py`: masa, fricción, torque máximo, ruido en θ y θ̇, y latencia de 0–2 pasos.
- `configs/env/level_{1,2,3}.yaml`, con el mismo espacio de observación en todos los niveles.
- Todo lo aleatorio usa `self.np_random`. Cada componente lleva sus pruebas.
- El PID se vuelve a evaluar en los niveles 1–3, con el término de giro activado para los niveles 2 y 3.

## Decisiones que afectan esta fase

| Decisión | Cómo se maneja en F2 |
|---|---|
| D4 — Presupuesto de pasos | Se mide en T5/T6 y se propone en T7 |
| D5 — TensorBoard / W&B | Se empieza con TensorBoard, que ya está instalado. Si se quiere W&B para comparar entre las dos máquinas, se decide antes de F3 |
| D8 — Normalización de observaciones (`VecNormalize`) | Se prueba en F2 con y sin normalizar. `train.py` la expone como opción `normalize` en `configs/training/common.yaml`, igual para los 4 algoritmos |
