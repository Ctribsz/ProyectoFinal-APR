# Guion del video final

Duración objetivo: **unos 8 minutos**. Habla quien aparece en cada bloque: **C** es Christian y **M** es Mario.
La columna *Pantalla* dice qué se graba; la diapositiva es la de la presentación.

El texto es una guía para decirlo con sus palabras, no para leerlo palabra por palabra.

## Antes de grabar

- Presentación en modo *Present*: los videos de las diapositivas 6 y 7 se reproducen solos. La presentación tiene 10 diapositivas.
- Videos a mano, para grabarlos en pantalla completa:
  - `results/videos/caida_libre.mp4`;
  - `results/videos/pid_balanceo.mp4`;
  - `results/videos_finales/mosaico_nivel2.mp4` (ya no está en la presentación).
- Una terminal en la raíz del repo con el entorno activado (`source .venv/bin/activate`).
- El editor abierto en:
  - `assets/balance_bot.xml`;
  - `configs/env/level_3.yaml`;
  - `configs/algos/td3.yaml`.

## Bloques

| # | Tiempo | Quién | Pantalla |
|---|---|---|---|
| 1 | 0:00–0:30 | C | Diapositiva 1 |
| 2 | 0:30–1:10 | C | `caida_libre.mp4` y diapositiva 2 |
| 3 | 1:10–1:40 | M | Diapositiva 2 (hipótesis) |
| 4 | 1:40–2:50 | C | Editor y `pid_balanceo.mp4`; diapositiva 3 |
| 5 | 2:50–3:50 | M | Diapositiva 4, editor y terminal |
| 6 | 3:50–4:30 | M | Diapositiva 5, gráfica izquierda |
| 7 | 4:30–5:40 | C | `mosaico_nivel2.mp4` y diapositiva 6 |
| 8 | 5:40–6:10 | C | Diapositiva 7 |
| 9 | 6:10–7:00 | M | Diapositiva 8 y gráfica derecha de la 5 |
| 10 | 7:00–7:30 | M | Diapositiva 9, arriba |
| 11 | 7:30–8:00 | C | Diapositiva 9, abajo, y la 10 |

### 1. Presentación (C, 0:00–0:30)

> Hola, somos Christian Echeverría y Mario Betancourt. En este proyecto entrenamos un robot de dos ruedas para
> que se mantenga en equilibrio usando aprendizaje por refuerzo profundo. Comparamos cuatro algoritmos, DDPG,
> TD3, SAC y PPO, entre ellos y contra un controlador PID clásico.

### 2. El problema (C, 0:30–1:10)

*Pantalla: `caida_libre.mp4` y luego la diapositiva 2.*

> Así se comporta el robot sin control: se cae en menos de un segundo. Es un péndulo invertido sobre ruedas, y la
> única forma de mantenerlo vertical es mover las ruedas por debajo del centro de masa. Además, para avanzar
> primero tiene que retroceder un poco e inclinarse, lo que lo hace difícil de controlar. Y solo puede usar lo
> que mediría un robot real: una IMU y los encoders de las ruedas.
>
> Nuestra pregunta es qué algoritmo lo balancea mejor ante perturbaciones, y qué mecanismo de cada uno lo
> explica.

### 3. Hipótesis (M, 1:10–1:40)

*Pantalla: diapositiva 2, la lista de hipótesis.*

> Antes de los experimentos finales escribimos cuatro hipótesis. Que TD3 sobreestima menos el valor Q que DDPG y
> por eso es más estable. Que SAC se recupera mejor de empujones por su exploración con entropía. Que PPO
> necesita más pasos pero varía menos entre semillas. Y que los agentes igualan al PID en condiciones simples y
> lo superan en las difíciles.

### 4. Implementación: el entorno (C, 1:40–2:50)

*Pantalla:*
1. `assets/balance_bot.xml`, en la parte de los sensores.
2. `configs/env/level_3.yaml`.
3. `pid_balanceo.mp4`.
4. La diapositiva 3.

> El robot lo modelamos en MuJoCo, un motor de física para robótica. Aquí está el modelo: un chasis de un kilo
> sobre dos ruedas, con motores de torque limitado. Y estos son los sensores: la orientación y el giroscopio de
> la IMU, y la velocidad de cada rueda. La observación del agente sale solo de aquí, nunca de la posición real
> del simulador.
>
> El entorno tiene cuatro niveles de dificultad, que se configuran en estos archivos YAML: el nivel 0 es
> balancearse quieto, el 1 agrega empujones, el 2 agrega comandos de velocidad y giro, y el 3 agrega ruido en
> los sensores, latencia y física aleatoria. Entrenamos en el nivel 2 y evaluamos en todos.
>
> Como línea base hicimos un PID en cascada: aquí lo ven balanceando el robot en el nivel 0.

### 5. Implementación: entrenamiento y afinamiento (M, 2:50–3:50)

*Pantalla:*
1. Diapositiva 4.
2. `configs/algos/td3.yaml`.
3. En la terminal: `pytest -q` y, mientras corre, un entrenamiento corto (ver el comando abajo). Muestren unos segundos de cada uno.

```bash
python -m training.train --algo td3 --env-config configs/env/level_2.yaml --seed 99 --timesteps 20000 --out results/demo --no-diagnostics
```

> Para que la comparación sea justa, los cuatro algoritmos usan el mismo código de entrenamiento, la misma red y
> el mismo presupuesto; solo cambia este archivo de configuración. Los hiperparámetros se afinaron con Optuna:
> 40 pruebas de 200 mil pasos por algoritmo.
>
> Con la mejor configuración, cada algoritmo se entrenó un millón de pasos con 10 semillas. Fueron 60 corridas
> en total, repartidas entre la laptop, Google Colab y la computadora de Christian. Entre semillas reportamos la
> media intercuartil con intervalos de confianza por bootstrap. Todo el proyecto tiene 112 pruebas automáticas.

### 6. Resultados: aprendizaje (M, 3:50–4:30)

*Pantalla: diapositiva 5, la gráfica de la izquierda.*

> Estas son las curvas de aprendizaje. TD3, en naranja, llega al mejor retorno: 891 de 1000. SAC, en verde,
> aprende con menos muestras y es el más consistente entre semillas. DDPG, en azul, es el más lento e inestable.
> PPO necesita más pasos, pero cada paso es mucho más barato, así que en tiempo real es el más rápido.

### 7. Funcionamiento y generalización (C, 4:30–5:40)

*Pantalla: `mosaico_nivel2.mp4` en pantalla completa y luego la diapositiva 6 (tabla y video del nivel 3).*

> Aquí están los cinco controladores en el nivel de entrenamiento, con la misma semilla. De cada algoritmo
> mostramos la corrida mediana, no la mejor. Los cinco completan el episodio siguiendo los comandos.
>
> Lo interesante es qué pasa en condiciones que nunca vieron. Sin ruido, el PID es igual o mejor. Pero en el
> nivel 3, con ruido y latencia, el PID se cae en el 72 % de los episodios y todos los agentes lo superan. PPO es
> el que mejor generaliza, porque sus acciones son mucho más suaves y no amplifican el ruido.
>
> En este video del nivel 3 se ve: el PID se cae, SAC también, y DDPG, TD3 y PPO aguantan.

### 8. Prueba de empujón (C, 5:40–6:10)

*Pantalla: diapositiva 7.*

> También medimos cuánta fuerza aguanta cada uno con un empujón aislado. El PID tolera la mayor fuerza, 17
> newtons, pero tarda más de tres segundos en recuperarse. Los agentes se recuperan en alrededor de un segundo,
> y TD3 es el mejor en las dos métricas.

### 9. Aleatorización y diagnóstico de Q (M, 6:10–7:00)

*Pantalla: diapositiva 8 y luego vuelvan a la 5, a la gráfica de la derecha.*

> Después entrenamos directamente en el nivel 3, con aleatorización de dominio. DDPG, TD3 y SAC mejoran mucho en
> ese nivel; DDPG pasa de 283 a 654. A cambio aprenden una política más conservadora. PPO, en cambio, no se
> beneficia.
>
> Y este es el diagnóstico que explica la diferencia entre DDPG y TD3. Comparamos el valor Q que predice el
> crítico con el retorno que realmente se obtiene. DDPG sobreestima en 41 puntos; TD3, al tomar el mínimo de dos
> críticos, no sobreestima. Esa sobreestimación coincide con el entrenamiento inestable de DDPG.

### 10. Contraste de hipótesis (M, 7:00–7:30)

*Pantalla: diapositiva 9, la parte de las hipótesis.*

> Volviendo a las hipótesis: H1 se confirma. H2 se rechaza: SAC no se recupera mejor que TD3. H3 se cumple solo
> en parte, porque PPO no varía menos entre semillas. Y H4 se confirma con matices: el PID sigue siendo mejor
> ante empujones aislados.

### 11. Conclusión (C, 7:30–8:00)

*Pantalla: diapositiva 9, la parte de abajo, y luego la 10.*

> En conclusión, TD3 es el mejor algoritmo en las condiciones en que se entrena; SAC es la alternativa más
> confiable; y si el robot va a enfrentar ruido y latencia sin haberse entrenado para eso, PPO es la mejor
> opción. Frente al PID, el aprendizaje por refuerzo aporta sobre todo robustez ante ruido y mejor seguimiento de
> comandos.
>
> El código, el informe y los resultados están en el repositorio. Gracias.

## Consejos

- Graben cada bloque por separado y únanlos después; es más fácil repetir un bloque que todo el video.
- Si se pasan de tiempo, lo primero que se puede recortar es el bloque 9 (dejar solo el diagnóstico de Q) y el
  `pytest` del bloque 5.
- El entrenamiento corto tarda unos 2.5 minutos en la máquina de Christian. Lánzalo antes de grabar el bloque 5,
  muestra unos segundos de la salida y, si quieres, el resumen final (`retorno de evaluación: primero … → último …`).
  Con menos de 10 000 pasos falla al final porque todavía no hay evaluaciones. La salida queda en `results/demo/`,
  que no se versiona.
