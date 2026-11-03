# AGENTS.md — Guía para agentes de código

Este archivo lo leen los agentes de código (Claude Code u otros) antes de trabajar en el repositorio. Los humanos del equipo también lo siguen.

## Contexto en una línea

Robot de dos ruedas con auto-balanceo en MuJoCo; se comparan DDPG, TD3, SAC y PPO (Stable-Baselines3) contra un PID/LQR.

## Antes de hacer cualquier cosa

1. Leer `docs/especificacion-proyecto.md`. Es la fuente de verdad del proyecto.
2. Leer `docs/entregables.md` para saber qué entregable está activo y quién es responsable.
3. Leer el plan de la fase activa en `docs/` (actualmente `docs/fase-2-prueba-de-humo.md`).
4. Trabajar solo en tareas de la fase activa, salvo que se pida otra cosa explícitamente.

## Reglas de trabajo

### Cambios
- **Analizar antes de programar.** Antes de escribir código, explicar el enfoque en pocas líneas: qué archivos se tocan y por qué.
- **Proponer antes de modificar.** Para cambiar un archivo existente, describir el cambio y esperar aprobación. Archivos nuevos dentro de la tarea asignada no requieren aprobación previa.
- **Iterar, no reescribir.** Cambios acotados sobre el código existente. Nada de reescrituras completas de un archivo sin pedirlo.
- **Bitácora de cambios.** Al terminar una tarea, listar los cambios hechos: archivo, qué cambió y por qué.

### Especificación
- No cambiar la recompensa, los espacios de observación y acción ni los criterios de terminación sin actualizar primero las secciones 4.3–4.5 de la especificación.
- Si una decisión de la sección 13 de la especificación ("Decisiones abiertas") bloquea la tarea, detenerse y preguntar. No decidirla por cuenta propia.
- Cualquier cambio a la especificación se anota en su sección 14 ("Bitácora de cambios").

### Código
- Python 3.10+, con type hints en funciones públicas.
- Ningún hiperparámetro ni constante física va escrita directamente en el código: todo sale de `configs/` o del MJCF.
- Toda corrida guarda junto a sus resultados la semilla, la configuración YAML completa y las versiones de las librerías.
- Las observaciones solo pueden usar información medible por una IMU y encoders (ver especificación 4.3). Nunca la posición absoluta ni estados internos del simulador.
- Cada componente nuevo del entorno lleva una prueba en `tests/`.
- Correr `pytest` antes de dar una tarea por terminada.

### Lo que no se versiona
- `results/`, modelos entrenados (`*.zip` de SB3), bases de datos de Optuna (`*.db`), logs de TensorBoard y videos. Ya están en `.gitignore`.

## Comandos útiles

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.12: 3.14 aún no tiene todas las ruedas
pip install -r requirements.txt
pytest
python -c "from gymnasium.utils.env_checker import check_env; from envs.balance_bot_env import BalanceBotEnv; check_env(BalanceBotEnv(level=0))"
```

## Mapa del repositorio

| Carpeta | Contenido | Responsable principal |
|---|---|---|
| `assets/` | `balance_bot.xml` (MJCF) | Christian |
| `envs/` | `BalanceBotEnv`, perturbaciones, aleatorización | Christian |
| `baselines/` | PID/LQR | Christian |
| `configs/` | YAML de niveles y algoritmos | Christian; `algos/ddpg.yaml` y `algos/ppo.yaml`: Mario |
| `training/` | `train.py`, `tune.py`, `callbacks.py` | Christian |
| `evaluation/` | `evaluate.py`, `push_test.py` | Christian |
| `analysis/` | Notebooks, estadística con `rliable` | Ambos |
| `tests/` | Pruebas | Quien escriba el componente |

**Regla de reparto:** Mario (DDPG, PPO) y Christian (TD3, SAC) usan el mismo `train.py`, `tune.py` y `callbacks.py`. Los algoritmos se diferencian solo por su YAML en `configs/algos/`. Nadie crea un pipeline paralelo, porque rompería la comparación justa (especificación §6).
