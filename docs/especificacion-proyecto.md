# Proyecto final CC3104 — Robot de auto-balanceo con DRL

> Documento vivo. Es la fuente de verdad del proyecto para el equipo y para cualquier agente de código.
> Todo lo marcado como **[PROPUESTA]** es un valor inicial razonable, no una decisión cerrada.
> Todo lo marcado como **[ABIERTO]** está pendiente de decidir (ver sección 13).

---

## 1. Resumen

Entrenar un robot de dos ruedas tipo péndulo invertido, simulado en MuJoCo, para que mantenga el equilibrio vertical de forma autónoma, siga comandos de velocidad y se recupere de perturbaciones externas.

Se comparan cuatro algoritmos de control continuo usando implementaciones de Stable-Baselines3 (SB3):

- **DDPG**
- **TD3**
- **SAC**
- **PPO**

Además, se usa un controlador clásico (PID o LQR) como línea base de referencia.

La pregunta central del proyecto:

> ¿Qué algoritmo ofrece el mejor desempeño para el auto-balanceo robusto ante perturbaciones, y qué mecanismos técnicos de cada algoritmo explican ese resultado?

### Qué hace que esto no sea "correr `model.learn()` cuatro veces"

1. **Entorno propio** con niveles de dificultad crecientes (sección 4.6).
2. **Comparación justa**: mismo presupuesto de afinamiento y de pasos para todos, con varias semillas (secciones 6 y 7).
3. **Diagnósticos internos** que explican el resultado: sobreestimación de Q, entropía, KL, suavidad de acciones (sección 8).
4. **Línea base clásica** que demuestra si el RL aporta algo frente al control tradicional.

---

## 2. Alcance

### Dentro del alcance
- Modelo físico del robot en MJCF.
- Entorno `gymnasium.Env` con niveles de dificultad y sistema de perturbaciones.
- Controlador PID o LQR de referencia.
- Afinamiento de hiperparámetros con Optuna para los 4 algoritmos.
- Experimentos finales con varias semillas, pruebas de robustez y análisis estadístico.
- Informe técnico con dictamen.

### Fuera del alcance (salvo que sobre tiempo)
- Implementar los algoritmos desde cero.
- Navegación, mapas, evasión de obstáculos.
- Despliegue en un robot físico (ver sección 11: el diseño lo deja posible, pero no es obligatorio).

---

## 3. Stack técnico

| Componente | Uso |
|---|---|
| Python 3.10+ | Lenguaje base |
| `mujoco` | Motor de física (bindings oficiales) |
| `gymnasium` | Interfaz del entorno |
| `stable-baselines3` | DDPG, TD3, SAC, PPO |
| `rl_zoo3` + `optuna` | Afinamiento de hiperparámetros |
| `rliable` | Estadística robusta entre semillas (IQM, intervalos bootstrap) |
| `tensorboard` o `wandb` | Registro de métricas |
| `numpy`, `pandas`, `matplotlib` | Análisis y gráficas |
| `pyyaml` | Configuración de experimentos |

No se requiere GPU: las políticas son MLP pequeñas y MuJoCo corre rápido en CPU. Se paraleliza con `SubprocVecEnv`.

---

## 4. Especificación del entorno

### 4.1 Modelo físico (MJCF)

Robot de dos ruedas coaxiales con un cuerpo que pivota sobre el eje de las ruedas.

| Parámetro | Valor inicial | Estado |
|---|---|---|
| Altura del cuerpo | 0.30 m | [PROPUESTA] |
| Masa del cuerpo | 1.0 kg | [PROPUESTA] |
| Radio de rueda | 0.04 m | [PROPUESTA] |
| Masa por rueda | 0.05 kg | [PROPUESTA] |
| Separación entre ruedas | 0.18 m | [PROPUESTA] |
| Torque máximo por motor | 0.3 N·m | [PROPUESTA] |
| Paso de simulación | 0.002 s | [PROPUESTA] |
| `frame_skip` | 5 (control a 100 Hz) | [PROPUESTA] |
| Fricción rueda-suelo (deslizamiento) | 1.5 | [PROPUESTA] — con 1.0 las ruedas patinan a torque máximo |
| Amortiguamiento del eje de rueda | 0.001 N·m·s/rad | [PROPUESTA] — fricción viscosa del motor |
| Integrador | `implicitfast` | [PROPUESTA] |

**Convención de ejes** (fijada en `assets/balance_bot.xml` y verificada en `tests/test_model.py`):
- Marco del chasis: x hacia adelante, y sobre el eje de las ruedas (rueda izquierda en +y), z hacia arriba.
- Pitch θ es la rotación alrededor de +y. **θ > 0 es inclinarse hacia adelante.**
- Velocidad de rueda > 0 hace avanzar el robot hacia +x.
- Torque > 0 en ambas ruedas: el robot acelera hacia +x y el chasis se inclina hacia atrás (θ disminuye). Para corregir una caída hacia adelante se manda torque positivo.
- Yaw rate ψ̇ > 0 (antihorario visto desde arriba) cuando la rueda derecha gira más rápido que la izquierda.

**Nota sobre el torque máximo:** con 0.3 N·m y radio de 0.04 m, la aceleración máxima es de unos 13.6 m/s², mayor que g. A torque máximo desde el reposo, el chasis rota hacia atrás tan rápido que las ruedas se despegan del suelo y el robot pasa |θ| > 0.6 en menos de 0.1 s. Es físicamente correcto con estos valores y favorece políticas de control brusco, lo que se verá en la métrica de suavidad (§7.4). Si resulta problemático, se revisa con D1 (modelo de motor DC) o D3 (dimensiones).

Si en el futuro se quiere un robot físico, estos valores se reemplazan por los medidos del hardware real (sección 11).

**Movimiento: 3D con yaw [PROPUESTA].** Con dos motores independientes el robot puede girar, lo que permite el nivel de seguimiento de comandos de giro.

### 4.2 Espacio de acción

- `Box(low=-1, high=1, shape=(2,))`: comando normalizado para motor izquierdo y derecho.
- Se escala a torque: `torque = a * TORQUE_MAX`.
- **[ABIERTO]** Control por torque ideal vs. modelo de motor DC (voltaje → torque con constante del motor y fricción). El modelo de motor DC es más realista y facilita un posible paso al robot real.

### 4.3 Espacio de observación

**Regla de diseño:** solo se observa lo que un robot real podría medir con una IMU y encoders en las ruedas. No se usa la posición absoluta ni el estado interno del simulador.

| Índice | Observación | Fuente real equivalente |
|---|---|---|
| 0 | Ángulo de cabeceo (pitch) θ | IMU |
| 1 | Velocidad angular de cabeceo θ̇ | Giroscopio |
| 2 | Velocidad de giro (yaw rate) ψ̇ | Giroscopio |
| 3 | Velocidad angular rueda izquierda | Encoder |
| 4 | Velocidad angular rueda derecha | Encoder |
| 5 | Velocidad lineal comandada v* | Comando |
| 6 | Velocidad de giro comandada ψ̇* | Comando |
| 7–8 | Acción anterior (2 valores) | Memoria del controlador |

En los niveles con ruido, a las observaciones 0–4 se les suma ruido gaussiano (sección 4.6).

**Límites de la observación** (iguales en todos los niveles, porque se entrena en uno y se evalúa en otros):

| Índices | Límite | Origen |
|---|---|---|
| 0 | ±π/2 rad | Rango del pitch calculado desde el cuaternión |
| 1–2 | ±34.9 rad/s | `cutoff` del giroscopio en el MJCF (±2000 °/s, rango típico de una MPU6050) |
| 3–4 | ±100 rad/s | `cutoff` de los encoders en el MJCF |
| 5 / 6 | ±`v_max` / ±`yaw_rate_max` | `commands` en el YAML del nivel (0.5 m/s y 1.0 rad/s [PROPUESTA]) |
| 7–8 | ±1 | Rango de la acción |

Las velocidades de rueda (3–4) son relativas al chasis, como las mide un encoder. La velocidad lineal que usa la recompensa se estima con sensores: `v = R · ((ω_izq + ω_der)/2 + θ̇)`.

### 4.4 Recompensa [PROPUESTA]

Recompensa por paso, como suma de componentes con pesos configurables en YAML:

```
r_t = w_alive
    - w_theta  * θ²
    - w_dtheta * θ̇²
    - w_vel    * (v - v*)²
    - w_yaw    * (ψ̇ - ψ̇*)²
    - w_effort * ||a_t||²
    - w_smooth * ||a_t - a_{t-1}||²
```

| Peso | Valor inicial |
|---|---|
| `w_alive` | 1.0 |
| `w_theta` | 1.0 |
| `w_dtheta` | 0.1 |
| `w_vel` | 0.5 (0 en niveles sin comandos) |
| `w_yaw` | 0.2 (0 en niveles sin comandos) |
| `w_effort` | 0.01 |
| `w_smooth` | 0.05 |

**Regla:** cada componente se registra por separado en los logs para poder diagnosticar qué está optimizando realmente el agente.

### 4.5 Terminación y truncamiento

- **Terminación** (`terminated=True`): `|θ| > 0.6 rad` (unos 35°). El robot se considera caído.
- **Truncamiento** (`truncated=True`): 1000 pasos (10 s a 100 Hz).
- **Estado inicial:** θ uniforme en [-0.05, 0.05] rad, velocidades en cero.

### 4.6 Niveles de dificultad

| Nivel | Nombre | Comandos | Perturbaciones | Ruido / aleatorización |
|---|---|---|---|---|
| 0 | Balanceo en reposo | v* = 0, ψ̇* = 0 | Ninguna | Ninguno |
| 1 | Balanceo con empujones | v* = 0, ψ̇* = 0 | Impulsos aleatorios | Ninguno |
| 2 | Seguimiento de velocidad | v* y ψ̇* cambian cada 2–4 s | Impulsos aleatorios | Ninguno |
| 3 | Robusto | Igual que nivel 2 | Impulsos aleatorios | Ruido en sensores, latencia, masa y fricción aleatorias |

**Perturbaciones [PROPUESTA]:** fuerza horizontal aplicada al cuerpo con `data.xfrc_applied` durante 0.1 s, en instantes aleatorios (proceso de Poisson, en promedio una cada 2 s). La magnitud se muestrea de U(0, F_max), con F_max configurable por nivel.

Detalles fijados en E8 (implementación en `envs/perturbations.py`):
- **Punto de aplicación:** site `push_point`, a 0.27 m sobre el eje de las ruedas, en la parte alta del chasis. Si se aplica en el centro de masa, el robot se traslada rodando sin inclinarse y el empujón no pone a prueba el balanceo: el PID toleraba 84 N.
- **Dirección:** `sagittal` (hacia adelante o atrás del robot) por defecto. Un empujón lateral vuelca el robot sobre el eje de las ruedas y ningún controlador puede corregirlo: el PID tolera solo 4.8 N de lado. `direction: any` queda disponible como opción.
- **F_max = 8 N** en los niveles 1–3. Como referencia, el PID tolera 17.3 N con un solo empujón a t = 2 s. Con 15 N, los empujones repetidos llevan al robot a ~2.4 m/s y combinados con giros lo vuelcan de lado.
- Los empujones programados (`perturbations.schedule`) sirven para la prueba de empujón de §7.4.

**Aleatorización del nivel 3 [PROPUESTA]:**

| Parámetro | Rango |
|---|---|
| Masa del cuerpo | ±20 % |
| Fricción de ruedas | ±30 % |
| Ruido en θ | σ = 0.01 rad |
| Ruido en θ̇ | σ = 0.05 rad/s |
| Latencia de observación | 0–2 pasos |
| Torque máximo | ±10 % |
| Ruido en ψ̇ | σ = 0.05 rad/s [PROPUESTA] |
| Ruido en encoders | σ = 0.1 rad/s [PROPUESTA] |

Detalles fijados en E8 (`envs/randomization.py`):
- Masa, fricción, torque y latencia se muestrean al inicio de cada episodio; el ruido, en cada paso.
- La fricción de un contacto en MuJoCo es el máximo de las dos geometrías, así que se escala la de las ruedas y la del suelo juntas.
- La inercia del chasis se escala junto con su masa.
- **La recompensa y la terminación usan las lecturas limpias.** El ruido y la latencia solo afectan lo que observa el agente.

Cada nivel se selecciona con un parámetro del constructor: `BalanceBotEnv(level=2)`.

---

## 5. Línea base clásica

- **[ABIERTO]** PID en cascada (lazo interno de ángulo y lazo externo de velocidad) o LQR sobre el modelo linealizado.
- Se sintoniza a mano o por búsqueda simple sobre el nivel 0.
- Se evalúa con exactamente el mismo protocolo que los agentes de RL (sección 7).

**Criterio de validación del entorno:** si el PID no logra balancear el robot en el nivel 0, el problema está en el modelo físico, no en el RL. No se entrena ningún agente hasta que esto funcione.

---

## 6. Protocolo de afinamiento (análisis de sensibilidad)

**Principio:** todos los algoritmos reciben el mismo presupuesto de afinamiento. Comparar con hiperparámetros por defecto no es válido.

| Parámetro | Valor | Estado |
|---|---|---|
| Nivel de entrenamiento para afinar | Nivel 2 | [PROPUESTA] |
| Pruebas de Optuna por algoritmo | 40 | [PROPUESTA] |
| Pasos por prueba | 200k | [PROPUESTA] |
| Muestreador / podador | TPE / Median pruner | [PROPUESTA] |
| Métrica objetivo | Retorno medio en evaluación | [PROPUESTA] |

### Hiperparámetros a variar

| Algoritmo | Hiperparámetros |
|---|---|
| DDPG | tasa de aprendizaje, ruido de exploración, τ (soft update), tamaño de batch |
| TD3 | tasa de aprendizaje, ruido de la política objetivo, retraso de actualización de la política, ruido de exploración |
| SAC | tasa de aprendizaje, coeficiente de entropía (fijo vs. automático), τ, tamaño de batch |
| PPO | tasa de aprendizaje, rango de clip, número de épocas, `n_steps`, λ de GAE |

Arquitectura de red fija para todos: MLP [256, 256] **[PROPUESTA]**, para que la red no sea una variable más de la comparación.

Las corridas de Optuna se guardan completas: también son el material del análisis de sensibilidad (importancia de hiperparámetros, superficies de respuesta).

---

## 7. Protocolo experimental final

### 7.1 Entrenamiento
- Mejor configuración de cada algoritmo según la sección 6.
- **10 semillas** por algoritmo **[PROPUESTA]**, con un mínimo aceptable de 5.
- Presupuesto: 1M pasos de entorno por corrida **[PROPUESTA]**.
- Se entrena en el nivel 2 y se evalúa en los niveles 0–3. Entrenar en el nivel 3 es un experimento aparte (E5, sección 7.3).

### 7.2 Evaluación
- Política determinista (sin ruido de exploración).
- Semillas de evaluación fijas y distintas a las de entrenamiento.
- 50 episodios por nivel por semilla.

### 7.3 Experimentos

| Id | Experimento | Pregunta |
|---|---|---|
| E1 | Curvas de aprendizaje por pasos de entorno | ¿Quién aprende con menos interacción? |
| E2 | Curvas de aprendizaje por tiempo de reloj | ¿Quién aprende más rápido en la práctica? |
| E3 | Desempeño final en niveles 0–3 | ¿Quién generaliza a condiciones más duras? |
| E4 | Prueba de empujón | ¿Cuánta fuerza tolera cada uno y qué tan rápido se recupera? |
| E5 | Entrenamiento con aleatorización (nivel 3) vs. sin ella | ¿La aleatorización mejora la robustez? |
| E6 | Comparación contra PID/LQR | ¿Dónde aporta el RL frente al control clásico? |

### 7.4 Métricas

| Métrica | Definición |
|---|---|
| Retorno medio | Suma de recompensas por episodio |
| Tasa de caídas | % de episodios con terminación |
| Tiempo hasta caída | Pasos antes de `|θ| > 0.6` |
| Fuerza máxima tolerada | Mayor impulso a t = 2 s que no provoca caída (búsqueda binaria) |
| Tiempo de recuperación | Tiempo desde el impulso hasta que `|θ| < 0.035 rad` (unos 2°) de forma sostenida por 0.5 s |
| Error de seguimiento | RMSE de v y de ψ̇ respecto a los comandos |
| Suavidad | Media de `||a_t − a_{t−1}||` y energía de alta frecuencia de la señal de acción |
| Esfuerzo de control | Media de `||a_t||²` |
| Eficiencia de muestras | Pasos necesarios para alcanzar el 90 % del retorno final |

### 7.5 Estadística
- Se reporta el **IQM** (media intercuartil) con intervalos de confianza bootstrap al 95 %, usando `rliable`.
- No se reportan conclusiones basadas en una sola semilla.

---

## 8. Diagnósticos internos

Estos diagnósticos son los que permiten explicar *por qué* un algoritmo gana y no solo *cuál* gana.

| Diagnóstico | Algoritmos | Cómo se mide |
|---|---|---|
| Sobreestimación de Q | DDPG, TD3, SAC | Cada N pasos se toma un lote de estados y se compara Q(s, π(s)) del crítico con el retorno descontado real obtenido al ejecutar la política desde esos estados |
| Coeficiente de entropía | SAC | Evolución de α durante el entrenamiento y su relación con la recuperación tras empujones |
| KL aproximada y fracción de clip | PPO | Logs nativos de SB3 |
| Suavidad de acciones | Todos | Métrica de la sección 7.4, analizada por algoritmo |
| Componentes de la recompensa | Todos | Cada término de la sección 4.4 por separado |

Se implementan como `callbacks` de SB3 (`training/callbacks.py`, parámetros en `configs/training/diagnostics.yaml`). Solo los usan las corridas finales; el afinamiento no.

**Cómo se mide la sobreestimación de Q (E9).** El simulador no se puede reiniciar en un estado arbitrario a partir de la observación, que es parcial. Por eso se usa el método del artículo de TD3: cada 50k pasos se juegan 3 episodios deterministas, siempre con las mismas semillas, y en cada estado visitado se compara Q(s, π(s)) del primer crítico con el retorno descontado que la política obtuvo de verdad desde ese estado en ese episodio. En los episodios cortados por tiempo se descartan los últimos 300 pasos, ~3/(1 − γ), porque su retorno real quedó incompleto. En SAC, Q incluye el bono de entropía, así que parte de su sesgo positivo es por diseño. En esos mismos episodios se mide la suavidad (§7.4). La energía de alta frecuencia es la fracción del espectro de la acción por encima de 5 Hz, sin la componente de 0 Hz.

**Hipótesis a verificar** (se escriben antes de correr los experimentos y se contrastan después):
- H1: TD3 muestra menor sobreestimación de Q que DDPG y por eso entrena de forma más estable.
- H2: SAC se recupera mejor de empujones grandes por la exploración que le da la entropía.
- H3: PPO necesita más pasos de entorno, pero tiene menor varianza entre semillas.
- H4: los algoritmos de RL igualan al PID en el nivel 0 y lo superan en los niveles 2–3.

---

## 9. Estructura del repositorio [PROPUESTA]

```
balance-bot-drl/
├── README.md
├── docs/
│   └── especificacion-proyecto.md   ← este documento
├── assets/
│   └── balance_bot.xml              ← modelo MJCF
├── envs/
│   ├── balance_bot_env.py           ← BalanceBotEnv (gymnasium.Env)
│   ├── perturbations.py
│   └── randomization.py
├── baselines/
│   └── pid_controller.py
├── configs/
│   ├── env/level_{0..3}.yaml
│   └── algos/{ddpg,td3,sac,ppo}.yaml
├── training/
│   ├── train.py
│   ├── tune.py                      ← Optuna
│   └── callbacks.py                 ← diagnósticos (sección 8)
├── evaluation/
│   ├── evaluate.py
│   └── push_test.py
├── analysis/
│   └── notebooks/
├── results/                         ← no se versionan los modelos grandes
└── tests/
    └── test_env.py
```

---

## 10. Reparto y fases

### Reparto

| Persona | Responsabilidades |
|---|---|
| Christian | Modelo MJCF, `BalanceBotEnv`, niveles, perturbaciones, aleatorización, PID, `train.py`, `tune.py`, callbacks de diagnóstico, `evaluate.py`, `push_test.py`; afinamiento y corridas de **TD3 y SAC** |
| Mario | Configuración (`configs/algos/ddpg.yaml`, `ppo.yaml`), espacios de búsqueda, afinamiento, corridas con semillas y análisis de diagnósticos de **DDPG y PPO** |
| Ambos | Análisis estadístico, gráficas, informe y dictamen |

Los cuatro algoritmos usan las implementaciones de SB3 y pasan por el mismo `train.py`, `tune.py` y `callbacks.py`. Solo cambia su YAML. El detalle de entregables y fechas está en `docs/entregables.md`.

### Fases y criterios de aceptación

| Fase | Entregable | Criterio para pasar a la siguiente |
|---|---|---|
| F1 | Entorno + PID | `check_env` de gymnasium pasa; el PID balancea 10 s en el nivel 0; render visual correcto |
| F2 | Prueba de humo | Los 4 algoritmos con valores por defecto mejoran su retorno en el nivel 0 (el entorno se puede aprender) |
| F3 | Afinamiento | 40 pruebas por algoritmo completas y mejores configuraciones guardadas en `configs/algos/` |
| F4 | Experimentos finales | E1–E6 completos con todas las semillas |
| F5 | Análisis e informe | Hipótesis H1–H4 contrastadas; dictamen escrito |

---

## 11. Compatibilidad con un robot real (opcional)

No es obligatorio, pero el entorno se diseña para no cerrar esa puerta:

1. **Solo observaciones disponibles en hardware real** (IMU + encoders). Ya aplicado en la sección 4.3.
2. **Modelo de motor DC** en lugar de torque ideal (decisión abierta en la sección 4.2).
3. **Latencia y ruido** modelados en el nivel 3.
4. **Frecuencia de control realista** (100 Hz).
5. **Parámetros físicos medidos** del hardware real si se construye (masas, dimensiones, constantes del motor).
6. **Política pequeña** (MLP) exportable a ONNX o a pesos en C para un microcontrolador.

---

## 12. Convenciones para agentes de código

- Este documento es la fuente de verdad. Si un cambio de código contradice la especificación, primero se actualiza este documento y se anota en la bitácora de cambios (sección 14).
- Toda corrida debe ser reproducible: semilla explícita, configuración YAML guardada junto a los resultados y versión de librerías registrada.
- No cambiar la función de recompensa, los espacios de observación/acción ni los criterios de terminación sin actualizar las secciones 4.3–4.5.
- Ningún hiperparámetro va escrito directamente en el código: todo sale de `configs/`.
- Cada componente nuevo del entorno lleva una prueba en `tests/`.
- Antes de modificar un archivo existente, describir el cambio propuesto y esperar aprobación.

---

## 13. Decisiones abiertas

| # | Decisión | Opciones | Notas |
|---|---|---|---|
| D1 | Control del motor | Torque ideal / modelo de motor DC | El modelo DC ayuda si se quiere robot real |
| D2 | Línea base | **Resuelta (2026-11-02): PID en cascada** | Balancea 20/20 en el nivel 0; el LQR no hizo falta |
| D3 | Dimensiones del robot | Valores propuestos / medidos de un robot real | Depende de si se construye hardware |
| D4 | Presupuesto de pasos | **Resuelta (2026-11-09):** 1M de pasos por corrida; 10 semillas en el nivel 2 y 5 en el nivel 3 (E5), iguales para los 4 algoritmos; afinamiento de 40 pruebas × 200k pasos en el nivel 2 para los 4 | PPO tiene 10 semillas en el nivel 3, pero la comparación usa las semillas 0–4 |
| D5 | Plataforma de registro | TensorBoard / W&B | W&B facilita comparar entre los dos integrantes |
| D6 | Formato de entrega | Por confirmar con el catedrático | |
| D8 | Normalización de observaciones (`VecNormalize`) | **Resuelta (2026-11-09): sin normalizar**, igual para los 4 | La prueba de F2 con una semilla no fue concluyente y no hubo tiempo de repetirla; se declara como limitación en el informe |
| D7 | Fecha de entrega | **Resuelta:** avance el 2026-11-02; entrega final el 2026-11-09 (se adelantó desde el ~13) | Cronograma en `docs/entregables.md` |

---

## 14. Bitácora de cambios

| Fecha | Cambio | Autor |
|---|---|---|
| 2026-10-19 | Versión inicial del documento | Equipo |
| 2026-11-09 | §13: D4 resuelta (1M de pasos; 10 semillas en el nivel 2 y 5 en el nivel 3; afinamiento 40 × 200k para los 4), D8 resuelta (sin normalizar) y D7 actualizada (entrega el 2026-11-09). §7.2–§7.4 (E11): semillas de evaluación final = 300000 + k y de la prueba de empujón = 400000 + k; la prueba de empujón se hace en el nivel 0, adelante y atrás, con 6 s de simulación tras el empujón (el PID tarda ~3 s en recuperarse) y tiempo de recuperación medido a 4 y 8 N; parámetros en `configs/evaluation/evaluation.yaml`. Excepción a AGENTS.md: `results/final` (corridas de E12) y `results/evaluation` se versionan con `git add -f` para compartirlos | Christian |
| 2026-11-06 | §8 (E9): método de la sobreestimación de Q (episodios deterministas, Q(s, π(s)) contra el retorno descontado real; se descartan los últimos 300 pasos de episodios truncados) y definición de la energía de alta frecuencia (> 5 Hz). Los diagnósticos solo corren en las corridas finales | Mario |
| 2026-11-02 | §4.6 (E8): punto y dirección de los empujones, F_max = 8 N, ruido en ψ̇ y encoders, detalles de la aleatorización; la recompensa usa lecturas limpias | Christian |
| 2026-11-02 | Cierre de F1. §5: el PID en cascada balancea 20/20 en el nivel 0; ganancias en `configs/baselines/pid.yaml`. D2 resuelta: PID en cascada. §13: nueva decisión abierta D8 (normalización de observaciones) | Christian |
| 2026-11-01 | §4.3: límites finitos de la observación (saturación de los sensores en el MJCF y rangos de comando en el YAML) y estimación de v con sensores (resultado de E2) | Christian |
| 2026-11-01 | §4.1: convención de ejes, fricción rueda-suelo 1.5, amortiguamiento de rueda 0.001, integrador `implicitfast` y nota sobre el torque máximo (resultado de E1) | Christian |
| 2026-11-01 | §10: reparto por algoritmo (Mario: DDPG y PPO; Christian: el resto). §13: D7 resuelta. §7.1: redacción corregida. Se agrega `docs/entregables.md` | Christian |
