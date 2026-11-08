# Reparto de las corridas finales (E12)

Quién corre cada semilla y dónde. Sirve para que **ninguna semilla se corra dos veces** y para saber qué falta.
Se actualiza al lanzar y al terminar cada grupo de semillas.

## Reglas

1. **Mismo código en todas las máquinas:** el commit de `main` con las configuraciones afinadas de los 4
   algoritmos (E10). Se anota en la columna *Commit*; las corridas lo guardan en `meta.json` (con git) y
   `analysis/finales.py` lo muestra.
2. **Cada semilla en una sola máquina.** Si una se repite, `analysis/finales.py` lo avisa.
3. **Mismo número de semillas para los 4 algoritmos dentro de un bloque** (§6). Entre bloques puede variar.
   Primero se completan las semillas 0–4 (mínimo de §7.1); las 5–9, si alcanza el tiempo. Decisión D4.
4. **E2 (tiempo de reloj)** solo compara algoritmos que corrieron en la misma máquina.

Estados: *pendiente*, *corriendo*, *terminada*, *copiada* (ya está en `results/final` de la máquina del análisis).

## Bloque 1: entrenamiento en el nivel 2 (§7.1)

| Algoritmo | Semillas | Máquina | Responsable | Commit | Inicio | Estado |
|---|---|---|---|---|---|---|
| DDPG | 0–3 | Laptop de Mario (LAPTOP-V9KBFJLE), 4 a la vez | Mario | | | pendiente |
| DDPG | 4–5 | Colab de Mario | Mario | | | pendiente |
| DDPG | 6–9 | Otras PC (por definir) | Mario | | | pendiente |
| PPO | 0–9 | Laptop de Mario (pendiente de afinar) | Mario | | | pendiente |
| TD3 | 0–9 | | Christian | | | pendiente |
| SAC | 0–9 | | Christian | | | pendiente |

## Bloque 2: entrenamiento en el nivel 3 (experimento E5, §7.3)

| Algoritmo | Semillas | Máquina | Responsable | Commit | Inicio | Estado |
|---|---|---|---|---|---|---|
| DDPG | 0–9 | | Mario | | | pendiente |
| PPO | 0–9 | | Mario | | | pendiente |
| TD3 | 0–9 | | Christian | | | pendiente |
| SAC | 0–9 | | Christian | | | pendiente |

## Comandos

En la laptop o en otra PC con el repo (`lab\instalar.bat` instala todo), en cmd dentro de la carpeta del repo:

```bat
set PYTHONUTF8=1
.venv\Scripts\python.exe -m lab.semillas --algo ddpg --level 2 --seeds 0-3 --workers 4
```

- `--workers`: corridas a la vez; el número de núcleos físicos de la PC.
- Si se corta, el mismo comando salta las terminadas y repite la que quedó a medias.

En Colab: notebook `lab/colab.ipynb`, celdas 1–4 y luego 8–9 (`ALGO_E12`, `NIVEL`, `SEMILLAS`).

Al terminar, se copian a `results/final/` de la máquina del análisis solo las carpetas `<algo>_level<n>_seed<s>`
con `meta.json`, y luego se corre `python -m analysis.finales`.

## Tiempo aproximado por semilla (1M de pasos)

| | DDPG | TD3 / SAC | PPO |
|---|---|---|---|
| Laptop de Mario, 4 a la vez | ~8 h | ~8 h (estimado) | ~2 h (~40 min si corre sola) |
| Colab gratis, 1 a la vez | ~3.3 h | ~3.3 h (estimado) | ~30 min |
| Máquina de Christian | ~1.4 h | ~1.4 h (estimado) | ~10 min |

Las cifras de DDPG y PPO salen de las mediciones del 2026-11-06 y del afinamiento; las de TD3 y SAC suponen un
costo parecido al de DDPG (también entrenan la red en cada paso).
