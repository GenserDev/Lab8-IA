"""
Ejercicio 4 – Localización del fantasma de Pac-Man (tablero 10x10)

CSV: Sensor_Color_Distribution.csv
  - Columnas: dist, R, O, Y, G, B
  - Cada columna representa P(distancia | color): suma 1 por columna
  - Likelihood para actualización bayesiana: P(dist(s,G) | color_observado)

Modelo:
  - Prior uniforme: P(G = celda) = 1/100
  - Actualización: P(G | evidencia) ∝ P(dist | color) * P(G | evidencia previa)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

GRID = 10
CSV_PATH = os.path.join(os.path.dirname(__file__), "Sensor_Color_Distribution.csv")

# Columnas tal como aparecen en el CSV
COLORS = ["R", "O", "Y", "G", "B"]
COLOR_NAMES = {"R": "Red", "O": "Orange", "Y": "Yellow", "G": "Green", "B": "Blue"}
COLOR_HEX = {
    "R": "#e74c3c",
    "O": "#e67e22",
    "Y": "#f1c40f",
    "G": "#2ecc71",
    "B": "#3498db",
}

rng = np.random.default_rng(42)


def load_sensor_dist(path):
    # index = distancia Manhattan (0-18), columnas = R O Y G B
    return pd.read_csv(path, index_col="dist")


def manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)


def likelihood_grid(sensor_dist, sensor_r, sensor_c, color):
    """P(dist(sensor, ghost) | color) para cada celda posible del fantasma."""
    grid = np.zeros((GRID, GRID))
    max_d = len(sensor_dist) - 1
    for gr in range(GRID):
        for gc in range(GRID):
            d = min(manhattan(sensor_r, sensor_c, gr, gc), max_d)
            grid[gr, gc] = sensor_dist.loc[d, color]
    return grid


def bayes_update(belief, sensor_dist, sensor_r, sensor_c, color):
    lk = likelihood_grid(sensor_dist, sensor_r, sensor_c, color)
    posterior = belief * lk
    posterior /= posterior.sum()
    return posterior


def sample_color(sensor_dist, ghost_r, ghost_c, sensor_r, sensor_c):
    """Muestrea un color dado la distancia real al fantasma.
    Convierte P(dist|color) → P(color|dist) normalizando la fila."""
    d = min(manhattan(sensor_r, sensor_c, ghost_r, ghost_c), len(sensor_dist) - 1)
    row = sensor_dist.loc[d, COLORS].values.astype(float)
    row_sum = row.sum()
    probs = row / row_sum if row_sum > 0 else np.ones(len(COLORS)) / len(COLORS)
    return rng.choice(COLORS, p=probs)


def plot_belief(ax, belief, title, ghost_pos=None, sensor_pos=None, color=None):
    im = ax.imshow(belief, cmap="YlOrRd", vmin=0, vmax=belief.max() + 1e-9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(GRID))
    ax.set_yticks(range(GRID))
    ax.tick_params(labelsize=6)

    if ghost_pos:
        ax.scatter(ghost_pos[1], ghost_pos[0], s=120, c="cyan",
                   marker="*", zorder=5, label="Ghost (real)")
    if sensor_pos and color:
        ax.scatter(sensor_pos[1], sensor_pos[0], s=80,
                   c=COLOR_HEX[color], marker="s", edgecolors="black",
                   linewidths=0.8, zorder=5, label=f"Sensor -> {COLOR_NAMES[color]}")
    if ghost_pos or sensor_pos:
        ax.legend(fontsize=6, loc="upper right")

    map_r, map_c = np.unravel_index(np.argmax(belief), belief.shape)
    ax.scatter(map_c, map_r, s=60, c="white", marker="D",
               edgecolors="black", linewidths=0.8, zorder=6)
    return im


def main():
    sensor_dist = load_sensor_dist(CSV_PATH)

    ghost_r, ghost_c = 3, 7   # posición real del fantasma (oculta al algoritmo)

    # Sensores colocados para triangular bien la posición del fantasma
    # (distancias Manhattan al fantasma: 4, 4, 3, 5, 4)
    sensor_positions = [(1, 5), (5, 9), (2, 9), (6, 5), (4, 4)]

    observations = [
        (r, c, sample_color(sensor_dist, ghost_r, ghost_c, r, c))
        for r, c in sensor_positions
    ]

    print("Observaciones simuladas (fantasma en ({},{})):".format(ghost_r, ghost_c))
    for r, c, col in observations:
        d = manhattan(r, c, ghost_r, ghost_c)
        print(f"  Sensor ({r},{c})  dist={d}  color={col}")

    belief = np.ones((GRID, GRID)) / (GRID * GRID)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle("Localización bayesiana del fantasma – Pac-Man 10×10", fontsize=12)

    # Panel 0: prior
    im = plot_belief(axes[0], belief, "Prior (uniforme)", ghost_pos=(ghost_r, ghost_c))
    plt.colorbar(im, ax=axes[0], fraction=0.04)

    for i, (sr, sc, color) in enumerate(observations):
        belief = bayes_update(belief, sensor_dist, sr, sc, color)
        title = f"Obs {i+1}: sensor ({sr},{sc}) → {color}"
        im = plot_belief(
            axes[i + 1], belief, title,
            ghost_pos=(ghost_r, ghost_c),
            sensor_pos=(sr, sc), color=color,
        )
        plt.colorbar(im, ax=axes[i + 1], fraction=0.04)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "ghost_tracking.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot guardado en: {out_path}")
    plt.show()

    # b) Estimación MAP
    map_r, map_c = np.unravel_index(np.argmax(belief), belief.shape)
    print("\n" + "=" * 45)
    print("b) Estimación MAP (máxima probabilidad a posteriori)")
    print("=" * 45)
    print(f"   Celda más probable : ({map_r}, {map_c})")
    print(f"   Probabilidad       : {belief[map_r, map_c]:.4f}")
    print(f"   Posición real      : ({ghost_r}, {ghost_c})")
    print()
    print("Top 5 celdas más probables:")
    flat_idx = np.argsort(belief.ravel())[::-1][:5]
    for rank, idx in enumerate(flat_idx, 1):
        r, c = divmod(int(idx), GRID)
        print(f"   {rank}. celda ({r},{c})  P = {belief[r,c]:.4f}")


if __name__ == "__main__":
    main()
