# Prueba de empujón (E4, H2)

Generado por `python -m evaluation.push_test`. Nivel 0, empujón único a t = 2 s en el punto alto del chasis, adelante y atrás, desde varios estados iniciales. Fuerza máxima: bisección (media por corrida). Recuperación: tiempo hasta |θ| < 2° sostenido 0.5 s (media de los episodios sin caída). Entre semillas: IQM con IC 95 %.

## Entrenados en el nivel 2

| Política | Fuerza máx. (N) | Recuperación a 4.0 N (s) | Recuperación a 8.0 N (s) | Caídas a 4.0 N | Caídas a 8.0 N |
|---|---|---|---|---|---|
| DDPG (10 semillas) | 10.5 [9.9, 12.0] | 0.96 [0.87, 1.29] | 1.45 [1.18, 1.72] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.25] |
| TD3 (10 semillas) | 12.7 [11.2, 13.6] | 0.78 [0.69, 0.92] | 1.06 [0.98, 1.20] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.08] |
| SAC (10 semillas) | 12.2 [11.7, 12.9] | 1.10 [0.91, 1.26] | 1.38 [1.27, 1.57] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| PPO (10 semillas) | 9.0 [8.5, 10.0] | 1.30 [1.16, 1.40] | 1.82 [1.70, 2.37] | 0.00 [0.00, 0.00] | 0.03 [0.00, 0.30] |
| PID | 17.2 | 2.58 | 3.35 | 0.00 | 0.00 |

## Entrenados en el nivel 3

| Política | Fuerza máx. (N) | Recuperación a 4.0 N (s) | Recuperación a 8.0 N (s) | Caídas a 4.0 N | Caídas a 8.0 N |
|---|---|---|---|---|---|
| DDPG (5 semillas) | 11.8 [11.3, 12.5] | 1.42 [1.23, 1.82] | 1.95 [1.53, 2.12] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| TD3 (5 semillas) | 8.8 [5.5, 9.4] | 1.12 [1.01, 1.46] | 1.43 [1.27, 2.09] | 0.00 [0.00, 0.33] | 0.33 [0.00, 0.83] |
| SAC (5 semillas) | 9.8 [9.4, 10.1] | 3.67 [2.62, 5.43] | 3.21 [3.04, 3.39] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| PPO (5 semillas) | 6.6 [4.9, 9.7] | 1.75 [1.47, 2.10] | 2.04 [2.03, 2.04] | 0.00 [0.00, 0.00] | 0.97 [0.33, 1.00] |
| PID | 17.2 | 2.58 | 3.35 | 0.00 | 0.00 |
