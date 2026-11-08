# Resumen de la sesión del 6 y 7 de noviembre (Mario con Claude)

Nota para retomar el trabajo en otra conversación o para ponerse al día con Christian. Antes de seguir, leer
`AGENTS.md`, `docs/especificacion-proyecto.md`, `docs/entregables.md` y `docs/reparto-e12.md`.

**Entrega final: lunes 9 de noviembre a medianoche.**

## 0. Actualización (domingo 8, 12:20)

- **E12, nivel 2, terminadas en la laptop:** DDPG 0–3 (retorno final 829, 871, 664 y 708) y PPO 0–4 (410, 746,
  684, 787 y 856). PPO llega a ~900 en todas las semillas, pero en varias cae al final: es el mismo desplome
  que en el afinamiento y conviene comentarlo en H3.
- **Corriendo:** DDPG 7–9 en la laptop (desde las 10:37, ~7 h), PPO 5–9 en la laptop (desde las 12:17, una a
  la vez) y DDPG 4–6 en Colab (el primer intento se desconectó).
- **Siguiente:** el bloque del nivel 3 (E5) cuando se libere la laptop (~18:00), juntar las corridas de Colab, y
  correr `python -m analysis.finales`.
- `docs/entregables.md` y `docs/reparto-e12.md` están al día con este estado.

## 1. Estado al cierre (sábado 7, ~22:45)

| Entregable | Estado |
|---|---|
| E9: `tune.py` + diagnósticos de §8 | **Listo.** Falta marcarlo en `entregables.md`. |
| E10: afinamiento de DDPG y PPO | **Listo.** Las configuraciones están en `configs/algos/ddpg.yaml` y `ppo.yaml`. |
| E10: afinamiento de TD3 y SAC | Christian. |
| E11: `evaluation/` | Christian. La carpeta todavía no existía. |
| E12: corridas finales | **Empezadas.** DDPG nivel 2, semillas 4–5, en Colab (desde ~21:10). PPO nivel 2, semillas 0–4, en la laptop (desde las 22:41; solo se verificó el arranque). |
| E13: análisis | `analysis/finales.py` listo; falta correrlo con los resultados reales. |

## 2. Mediciones de rendimiento (nivel 2, 1 hilo de torch)

| Pasos/s por proceso | DDPG | PPO |
|---|---|---|
| Laptop (i7-1065G7, 4 núcleos) | 58 | 410 |
| Colab gratis (1 núcleo real) | 83 | 632 |
| Máquina de Christian (nivel 0, E7) | ~200 | ~1946 |

- En Colab, 2 procesos solo dan +10 %: conviene 1.
- En la laptop, 4 procesos rinden ~35 pasos/s cada uno con DDPG y ~130 con PPO.
- **Por qué DDPG es más lento:** hace 1 actualización de gradiente con lotes de 256 en cada paso de entorno.
  PPO hace ~0.16 actualizaciones por paso con lotes de 64. Son unas 25 veces más muestras procesadas por paso.
  A cambio, DDPG reutiliza la experiencia (replay buffer) y en teoría necesita menos pasos. Esa diferencia es
  la que separan E1 (curvas por pasos) y E2 (curvas por tiempo de reloj).

## 3. Herramientas nuevas (todas en `main`)

| Archivo | Para qué |
|---|---|
| `lab/afinar.py` / `afinar.bat` | Afinamiento con varios procesos. **Se reanuda solo:** las pruebas a medias pasan a FAIL y no cuentan, lanza solo las que faltan y usa números de worker nuevos. Un `.lock` evita que se lance dos veces. |
| `lab/colab.py` + `lab/colab.ipynb` | Colab: afinar (celda 6) o correr semillas (celdas 8–9). Copia a Google Drive (`APR-colab/results/...`) y se reanuda al volver a correr las celdas. Cada comando entra solo a la carpeta del repo. |
| `lab/semillas.py` | Corridas finales por semilla (`--seeds 0-3 7`), varias a la vez, con nombres fijos `<algo>_level<n>_seed<s>`. Salta las terminadas, aparta las cortadas en `_incompletas/` y no duplica corridas vivas. |
| `training/callbacks.py` + `configs/training/diagnostics.yaml` | E9: sobreestimación de Q (DDPG, TD3, SAC) y suavidad (los 4), cada 50k pasos. Solo en las corridas finales; el afinamiento no los usa. |
| `analysis/resumen_afinamiento.py` | Resumen de cada estudio en `docs/afinamiento/<estudio>.md`, su CSV y su figura en `docs/img/`. |
| `analysis/finales.py` | E13: IQM y media con IC 95 % (bootstrap), E1, E2 (por máquina), H1 (Q), suavidad y pasos al 90 %. Avisa si hay menos de 5 semillas o semillas repetidas. |
| `docs/reparto-e12.md` | Qué semillas corre cada máquina. Se actualiza al lanzar y al terminar. |

**Cambios en archivos de Christian, conviene avisarle:**
- `training/train.py` acepta `diagnostics_cfg`.
- `training/train.py` anota el estado de git **al inicio** de la corrida, no al final (commit `6077539`). Antes,
  los commits hechos durante una corrida quedaban anotados como si fueran el código usado.
- `rliable.library` no funciona en el entorno: su dependencia `arch` no es compatible con pandas 3. `finales.py`
  usa `rliable.metrics` para el IQM y su propio bootstrap, con el mismo método por defecto de `rliable`.
  Hay que mencionarlo en el informe.

## 4. Resultados del afinamiento (E10)

Protocolo: 40 pruebas × 200k pasos en el nivel 2, 1 semilla por prueba, objetivo = media de las últimas
3 evaluaciones, TPE y poda por mediana. Detalle en `docs/afinamiento/`.

| | DDPG | PPO |
|---|---|---|
| Pruebas válidas | 40 (11 completas, 29 podadas) | 40 (22 completas, 18 podadas) |
| Duración | 7.6 h (4 procesos) | 2.6 h (4 procesos) |
| Mejor | #39, objetivo **836.3** | #34, objetivo **865.9** |
| Configuración elegida | lr 8.852e-4, σ 0.2357, τ 0.01708, batch 256 | lr 1.607e-4, clip 0.1628, n_epochs 16, n_steps 2048, λ 0.9498 |

Observaciones para el informe:
- **DDPG:** las mejores tienen más ruido de exploración (σ 0.19–0.34) y un τ mayor que los valores por
  defecto. La mejor saca 84 puntos a la segunda: puede haber ruido de semilla.
- **PPO:** hay una zona buena clara (4 pruebas entre 841 y 866). También **se desploma** con tasas de
  aprendizaje altas: la #21 tuvo un pico de 895 y terminó en 117. Esto importa para H3. En el nivel 2 tarda
  ~120k pasos en llegar a 800; DDPG, ~80k.
- **Limitación:** una semilla por prueba. No se revalidaron las mejores con más semillas, por falta de tiempo.

## 5. Semillas (D4): qué son y cuántas usar

- **Una semilla** fija todo el azar de una corrida: los pesos iniciales, la exploración, el orden de los
  lotes, y los estados iniciales, empujones, ruido y aleatorización del entorno. Cada semilla es una
  repetición independiente del experimento. La evaluación usa `semilla + 100000` y los diagnósticos
  `semilla + 200000`.
- **Por qué varias:** el resultado depende mucho de la semilla (en E7, DDPG terminó en 243 con una y en 70 con
  otra). §7.5 no permite conclusiones con una sola.
- **5 contra 10:** con 5, el IQM promedia las 3 del medio y los intervalos son anchos. Se ven las diferencias
  grandes, como PPO contra DDPG o RL contra el PID en el nivel 3, pero no las finas, como TD3 contra SAC.
  Con 10, el intervalo es ~30 % más angosto. El costo crece en proporción, y se paga dos veces (nivel 2 y E5).
- **Reglas:** el mismo número de semillas para los 4 algoritmos dentro de cada bloque. El número se decide por
  el tiempo disponible, no por si los resultados "salen significativos".
- **Plan acordado:** primero las semillas 0–4, que se necesitan en cualquier caso. El domingo en la tarde se
  decide si se agregan las 5–9 en el bloque del nivel 2. **Hay que cerrarlo con Christian.**

## 6. Plan de E12 (ver `docs/reparto-e12.md`)

| Bloque | Algoritmo | Semillas | Dónde | Estado |
|---|---|---|---|---|
| Nivel 2 | DDPG | 0–3 | Laptop, 4 a la vez (~8 h) | pendiente: cuando termine PPO |
| Nivel 2 | DDPG | 4–5 | Colab de Mario (~3.3 h cada una) | corriendo |
| Nivel 2 | DDPG | 6–9 | Otras PC (si se decide usar 10) | pendiente |
| Nivel 2 | PPO | 0–4 | Laptop, 4 a la vez (~2.7 h en total) | corriendo desde las 22:41 |
| Nivel 3 (E5) | DDPG, PPO | 0–4 | por repartir | pendiente |
| Ambos | TD3, SAC | | Christian | pendiente |

**Cuidado con E2:** el tiempo de reloj solo se compara dentro de una misma máquina. La laptop corre DDPG y PPO,
así que esa comparación sí vale.

## 7. Comandos frecuentes

Laptop, en **cmd**, dentro de `C:\Aprendizaje por Refuerzo\ProyectoFinal-APR`:

```bat
set PYTHONUTF8=1
.venv\Scripts\python.exe -m lab.semillas --algo ddpg --level 2 --seeds 0-3 --workers 4
.venv\Scripts\python.exe -m lab.afinar --algo ppo --workers 4
.venv\Scripts\python.exe -m analysis.resumen_afinamiento --algo ppo
.venv\Scripts\python.exe -m analysis.finales
.venv\Scripts\tensorboard.exe --logdir results\optuna\ddpg_level2
```

En **PowerShell**, el `set` es `$env:PYTHONUTF8 = '1'`.

**Colab:** abrir `lab/colab.ipynb` desde GitHub (rama `main`). Poner `RAMA = "main"` en la celda 1 y correr
las celdas 1–4. Para semillas, celda 8 (`ALGO_E12`, `NIVEL`, `SEMILLAS`) y celda 9. Si Colab ofrece
"Reiniciar sesión", no hace falta.

**Otras PC:** `git clone https://github.com/Ctribsz/ProyectoFinal-APR.git` (o *Download ZIP*), luego
`lab\instalar.bat` y `lab.semillas`. Usar tantos `--workers` como núcleos físicos tenga la PC. Traer de
vuelta solo las carpetas `results\final\<algo>_level<n>_seed<s>` que tengan `meta.json`.

## 8. Problemas encontrados y cómo se resolvieron

| Problema | Solución |
|---|---|
| Reanudar `lab.afinar` relanzaba 40 pruebas | Ahora cuenta las que faltan y marca las interrumpidas. |
| `afinar.bat` siempre usaba el worker 0 | Ya no pasa `--first-worker`. |
| Si el lanzador moría, otro lanzador podía duplicar una corrida aún viva | Cada corrida mantiene su propio `.lock`. |
| `meta.json` anotaba el commit del final de la corrida | Ahora anota el del inicio (`6077539`). |
| Colab: `No module named 'lab'` o `cd: No such file` | Celda 1 con `RAMA = "main"` (la rama vieja ya no existe) y comandos con `cd` al repo. |
| `$env:` en cmd | En cmd se usa `set PYTHONUTF8=1`. |
| `rliable.library` se rompe con pandas 3 | Bootstrap propio en `finales.py`. |

## 9. Pendientes

1. **Con Christian:** cerrar D4 (5 o 10 semillas), avisarle de los cambios en `train.py`, y preguntar cómo
   van su E10 y E11 y qué parte de su máquina queda libre para semillas.
2. Revisar que terminen bien las semillas 0–4 de PPO (laptop) y las 4–5 de DDPG (Colab), y copiar las de
   Colab a `results/final` de la laptop.
3. Lanzar DDPG 0–3 en la laptop cuando termine PPO, y repartir el bloque del nivel 3 (E5).
4. Actualizar `docs/entregables.md` (E9 y E10 listos) y `docs/reparto-e12.md` (estados y commits).
5. El lunes: correr `analysis.finales`, integrar E3, E4 y E6 desde `evaluation/` (E11) y escribir el informe.
   Las secciones de método y afinamiento ya se pueden escribir desde ahora.
6. Respaldar `results/` en Drive o en una USB: no se versiona.
