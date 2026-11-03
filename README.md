# Balance Bot DRL

Robot de dos ruedas con auto-balanceo, simulado en MuJoCo y controlado con aprendizaje por refuerzo profundo. El robot debe mantenerse vertical, seguir comandos de velocidad y recuperarse de empujones externos.

El proyecto compara cuatro algoritmos de control continuo (**DDPG, TD3, SAC y PPO**) contra un controlador clásico (PID/LQR), con afinamiento de hiperparámetros en igualdad de condiciones, varias semillas por configuración y diagnósticos internos que explican por qué cada algoritmo se comporta como lo hace.

> Proyecto final del curso CC3104 Aprendizaje por Refuerzo — Universidad del Valle de Guatemala, 2026.

---

## Pregunta de investigación

¿Qué algoritmo ofrece el mejor desempeño para el auto-balanceo robusto ante perturbaciones, y qué mecanismos técnicos de cada algoritmo explican ese resultado?

## Estado del proyecto

| Fase | Descripción | Estado |
|---|---|---|
| F1 | Entorno MuJoCo y controlador PID | Completa (2026-11-02) |
| F2 | Prueba de humo con los 4 algoritmos | En progreso |
| F3 | Afinamiento de hiperparámetros (Optuna) | Pendiente |
| F4 | Experimentos finales con semillas | Pendiente |
| F5 | Análisis e informe | Pendiente |

## El entorno

`BalanceBotEnv` es un entorno de `gymnasium` con cuatro niveles de dificultad:

| Nivel | Tarea |
|---|---|
| 0 | Mantener el equilibrio en reposo |
| 1 | Mantener el equilibrio con empujones aleatorios |
| 2 | Seguir comandos de velocidad y giro con empujones |
| 3 | Lo anterior con ruido en sensores, latencia y física aleatorizada |

El agente observa solo lo que un robot real podría medir: ángulo y velocidad angular de la IMU, velocidad de las ruedas y los comandos recibidos. Controla el torque de cada motor.

La especificación completa está en [`docs/especificacion-proyecto.md`](docs/especificacion-proyecto.md).

## Instalación

```bash
git clone https://github.com/Ctribsz/ProyectoFinal-APR.git
cd ProyectoFinal-APR
python3.12 -m venv .venv          # 3.10–3.12; 3.14 aún no tiene todas las ruedas
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # torch solo CPU (~200 MB en vez de ~2 GB)
pip install -r requirements.txt
```

## Uso

> Los comandos se completan conforme avanzan las fases.

```bash
# Evaluar el PID y grabar videos (F1)
python -m baselines.pid_controller --level 0 --episodes 20
python -m envs.recording --policy pid --seed 0 --out results/videos/pid.mp4
python -m mujoco.viewer --mjcf=assets/balance_bot.xml   # explorar el modelo

# Entrenar un agente (F2)
python -m training.train --algo sac --env-config configs/env/level_2.yaml --seed 0

# Afinar hiperparámetros (F3)
python -m training.tune --algo sac --trials 40

# Evaluar y prueba de empujón (F4)
python -m evaluation.evaluate --run results/<id-corrida>
python -m evaluation.push_test --run results/<id-corrida>
```

## Estructura

```
ProyectoFinal-APR/
├── assets/          modelo MJCF del robot
├── envs/            entorno, perturbaciones y aleatorización
├── baselines/       controlador PID/LQR
├── configs/         YAML de entornos y algoritmos
├── training/        entrenamiento, afinamiento y callbacks de diagnóstico
├── evaluation/      evaluación y prueba de empujón
├── analysis/        notebooks de análisis
├── results/         salidas de las corridas (no versionadas)
├── tests/           pruebas
└── docs/            especificación y planes por fase
```

## Resultados

Pendientes. Al cerrar la fase 4 esta sección tendrá la tabla comparativa, las curvas de aprendizaje y un video del robot recuperándose de perturbaciones.

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/especificacion-proyecto.md`](docs/especificacion-proyecto.md) | Especificación técnica completa: entorno, recompensa, protocolo experimental, métricas |
| [`docs/entregables.md`](docs/entregables.md) | Entregables, responsables y cronograma |
| [`docs/fase-1-entorno.md`](docs/fase-1-entorno.md) | Plan de trabajo de la fase actual con tareas y criterios de aceptación |
| [`AGENTS.md`](AGENTS.md) | Reglas para agentes de código que trabajen en el repositorio |

## Equipo

| Integrante | Carrera | Responsabilidades principales |
|---|---|---|
| Christian Echeverría | Ingeniería en Ciencias de la Computación | Modelo físico, entorno, PID, pipeline de entrenamiento y afinamiento, diagnósticos, evaluación, TD3 y SAC |
| Mario Betancourt | Ingeniería Mecatrónica | DDPG y PPO: configuración, afinamiento, corridas finales y análisis |

Ambos: análisis estadístico, informe y dictamen. Detalle en [`docs/entregables.md`](docs/entregables.md).
