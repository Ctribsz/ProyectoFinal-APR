# Uso de IA en el proyecto

Este documento registra cómo usamos inteligencia artificial durante el proyecto: en qué etapas, para qué y qué decisiones tomamos nosotros frente a lo que propuso la herramienta.

**Herramienta:** Claude (Anthropic), en conversación.
**Etapa cubierta:** definición del proyecto y documentación inicial del repositorio (15 de octubre al 1 de noviembre de 2026).

---

## Resumen

Usamos Claude como interlocutor técnico para escoger el proyecto, delimitar su alcance y escribir la documentación base del repositorio. Fue un proceso de ida y vuelta: nosotros planteábamos ideas y restricciones, Claude proponía opciones y señalaba riesgos, y nosotros decidíamos. El tema final del proyecto, el uso de librerías existentes y la decisión de no construir un robot físico fueron decisiones nuestras. Claude aportó sobre todo la estructura del protocolo experimental y la redacción de los documentos para agentes de código.

---

## Cómo fue la interacción

### 1. Lluvia de ideas

**Nosotros:** planteamos que el proyecto final debía basarse en lo último visto en clase (Deep Q-Learning y temas relacionados) y que su dificultad debía crecer con el número de integrantes. En ese momento éramos cinco.

**Claude:** propuso cinco opciones pensadas para repartirse entre cinco personas: semáforos inteligentes multiagente en SUMO, una ablación de Rainbow DQN, self-play en un juego de tablero, ajuste de un LLM pequeño con RL (GRPO) y manejo autónomo en `highway-env`. Para cada una señaló el riesgo principal y desaconsejó proyectos de trading con RL por la facilidad de sobreajustar al histórico.

### 2. Cambio de equipo

**Nosotros:** el equipo quedó en dos personas y preguntamos qué opciones seguían siendo viables.

**Claude:** redimensionó cada propuesta para dos personas, descartó la de LLM por el requerimiento de GPU y, a pedido nuestro, sugirió otras alternativas: autoscaling de servidores, gestión de inventario, manejo de una batería solar, generalización en MiniGrid y un estudio de *reward hacking*.

### 3. Elección del tema

**Nosotros:** escogimos un tema distinto a los propuestos: un robot de auto-balanceo simulado en MuJoCo, comparando DDPG, TD3, SAC y PPO, con análisis de sensibilidad y un estudio comparativo.

**Claude:** consideró el tema adecuado para dos personas, pero señaló varios riesgos:
- el balanceo puro lo resuelve bien un PID, así que los cuatro algoritmos podían terminar empatados;
- faltaba un controlador clásico como línea base;
- el entorno del robot había que construirlo, porque Gymnasium no lo trae;
- la comparación debía usar varias semillas y tener en cuenta que PPO es *on-policy* y los otros tres no.

**Qué cambió por esto:** incorporamos niveles de dificultad crecientes (empujones, seguimiento de velocidad, ruido y física aleatorizada) y un PID/LQR como referencia.

### 4. Uso de librerías existentes

**Nosotros:** confirmamos que podíamos usar implementaciones ya hechas de los algoritmos.

**Claude:** advirtió que, sin implementar los algoritmos, el proyecto corría el riesgo de verse como "correr `model.learn()` cuatro veces". Propuso tres formas de evitarlo, que adoptamos:
- dar a todos los algoritmos el mismo presupuesto de afinamiento con Optuna;
- poner el esfuerzo en el entorno y sus niveles;
- medir diagnósticos internos (sobreestimación de Q, entropía de SAC, suavidad de acciones) para explicar *por qué* gana un algoritmo, no solo *cuál*.

### 5. Especificación técnica y robot real

**Nosotros:** pedimos un documento inicial en Markdown y preguntamos qué tan viable era llevar esto a un robot físico.

**Claude:** redactó `docs/especificacion-proyecto.md`, con valores marcados como **[PROPUESTA]** y una lista de decisiones abiertas para que las cerráramos nosotros. Sobre el robot real, explicó los problemas típicos de pasar de simulación a la realidad y recomendó dejarlo fuera del alcance obligatorio, diseñando la simulación para no cerrar esa puerta.

**Nosotros:** comentamos que el proyecto nos sirve a ambos para el portafolio, porque uno de los dos es de mecatrónica. Claude propuso ampliar el alcance con el robot físico; **decidimos no hacerlo** y mantener el proyecto en simulación.

### 6. Documentación del repositorio

**Nosotros:** pedimos la documentación para subir al repositorio y empezar la implementación, siguiendo nuestra práctica de dejar documentación orientada a agentes de código.

**Claude:** generó:
- `README.md`: descripción, estado por fases, instalación y uso;
- `AGENTS.md`: reglas para agentes de código (leer la especificación primero, analizar antes de programar, proponer antes de modificar archivos existentes, no tocar recompensa ni espacios sin actualizar la especificación, cerrar cada tarea con una bitácora de cambios);
- `docs/fase-1-entorno.md`: plan de la primera fase con tareas, responsables, dependencias y criterios de salida;
- `requirements.txt` y `.gitignore`.

---

## Qué decidimos nosotros y qué aportó la IA

| Tema | Decisión del equipo | Aporte de Claude |
|---|---|---|
| Tema del proyecto | Robot de auto-balanceo en MuJoCo con DDPG, TD3, SAC y PPO | Evaluación de riesgos del tema |
| Implementación de algoritmos | Usar Stable-Baselines3 | Protocolo para que la comparación siga siendo rigurosa |
| Línea base clásica | Incluir PID/LQR | Propuesta inicial y criterio de validación del entorno |
| Niveles de dificultad | Adoptados | Diseño de los cuatro niveles |
| Robot físico | No incluirlo | Análisis de viabilidad y recomendación |
| Documentación para agentes | Estructura y reglas de trabajo | Redacción de los documentos |

---

## Cómo usamos y verificamos lo generado

- La especificación es un punto de partida: los valores marcados como **[PROPUESTA]** se validan en la fase 1 y cualquier cambio se anota en su bitácora.
- Las decisiones abiertas (modelo de motor, PID vs. LQR, presupuesto de pasos, entre otras) las tomamos nosotros conforme avanza el proyecto.
- El código que se escriba con ayuda de agentes sigue las reglas de `AGENTS.md` y se revisa y prueba (`pytest`, `check_env`) antes de integrarse.
- Somos responsables del contenido final del proyecto y del informe.

---

## Etapa 2: implementación (1 al 9 de noviembre de 2026)

**Herramienta:** Claude Code (Anthropic), el agente de programación que trabaja en la terminal sobre el repositorio.

Los dos trabajamos con la documentación de la etapa 1 como contrato con el agente: `docs/especificacion-proyecto.md`, `AGENTS.md` y el plan por entregables de `docs/entregables.md`. El agente proponía el siguiente paso según el plan; nosotros aprobábamos, corregíamos o cerrábamos las decisiones abiertas, que quedaron en la bitácora de la especificación (§14). Muchos de nuestros prompts fueron aprobaciones cortas ("VA sigamos con todo", "continua porfa").

Los prompts representativos, la evolución de la idea y cómo verificamos la solución están en la sección "Herramienta de IA generativa" del informe final (`docs/informe-final.pdf`). Un resumen de una sesión de Mario está en `docs/resumen-sesion-2026-11-07.md`.
