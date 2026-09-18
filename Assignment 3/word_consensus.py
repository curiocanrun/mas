import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.font_manager import FontProperties

from skimage.morphology import skeletonize


FONT = FontProperties(family="DejaVu Sans", weight="bold")

def _rasterize_letter(char, fontsize=260):
    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.patch.set_facecolor('white')
    ax.text(0.5, 0.42, char, fontsize=fontsize, ha='center', va='center',
             fontproperties=FONT, color='black')
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    gray = buf[..., :3].mean(axis=2)
    return gray < 128 

def _farthest_point_sampling(points, n, seed=0):
  
    rng = np.random.default_rng(seed)
    total = points.shape[0]

    if n >= total:
        return points.copy()
    selected = np.zeros(n, dtype=int)
    selected[0] = rng.integers(total)
    dist = np.sum((points - points[selected[0]])**2, axis=1)

    for i in range(1, n):
        selected[i] = np.argmax(dist)
        d = np.sum((points - points[selected[i]])**2, axis=1)
        dist = np.minimum(dist, d)

    return points[selected]

def letter_to_points(char, num_points=20, height=4.0):

    binary = _rasterize_letter(char)
    skeleton = skeletonize(binary)

    ys, xs = np.nonzero(skeleton)
    pts = np.column_stack([xs, -ys]).astype(float)  

    pts = _farthest_point_sampling(pts, num_points)

    pts = pts - pts.mean(axis=0)
    span = pts[:, 1].max() - pts[:, 1].min()
    if span > 0:
        pts = pts * (height / span)
    return pts

NAME = "CHARITHA" 
N = 20 

letters = [letter_to_points(ch, num_points=N) for ch in NAME]
letter_names = list(NAME)

p = 0.4

while True:
    G = nx.erdos_renyi_graph(N, p)
    if nx.is_connected(G):
        break

L = nx.laplacian_matrix(G).toarray()

eigvals = np.sort(np.linalg.eigvalsh(L))
print(f"Graph: N={N}, p={p}, edges={G.number_of_edges()}, \n"
      f"algebraic connectivity (lambda_2) = {eigvals[1]:.4f}")

dt = 0.05
steps_per_letter = 150 
history = [] 
titles = [] 
iter_in_letter = [] 

X = np.random.uniform(-6, 6, (N, 2))
X = X - np.mean(X, axis=0)

pinning_gains = np.zeros(N)
pinning_gains[0] = 1.0

A = nx.adjacency_matrix(G).toarray()

for idx, target in enumerate(letters):
    r = target
    for step in range(steps_per_letter):
        dX = np.zeros_like(X)
        for i in range(N):
            consensus_term = np.zeros(2)
            for j in range(N):
                if A[i, j] > 0:
                    consensus_term += A[i, j] * ((X[j] - X[i]) - (r[j] - r[i]))
            
            tracking_term = pinning_gains[i] * (r[i] - X[i])
            dX[i] = consensus_term + tracking_term

        X = X + dt * dX

        history.append(X.copy())
        titles.append(f"Forming: {letter_names[idx]}  ({idx + 1}/{len(letters)})")
        iter_in_letter.append(step)

margin = 5.0
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-margin, margin)
ax.set_ylim(-margin, margin)
ax.set_title("Multi-Agent Formation Control")
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_aspect('equal')

scatter = ax.scatter([], [], c='crimson', s=80, edgecolors='black', zorder=3)
edges_lines = [ax.plot([], [], 'gray', alpha=0.3, zorder=1)[0] for _ in G.edges()]

def init():
    scatter.set_offsets(np.empty((0, 2)))
    for line in edges_lines:
        line.set_data([], [])
    return [scatter] + edges_lines

def update(frame):
    current_X = history[frame]
    scatter.set_offsets(current_X)

    ax.set_title(
        f"{titles[frame]}\n"
        f"Iteration: {iter_in_letter[frame]}/{steps_per_letter - 1}   |   "
        f"Frame: {frame + 1}/{len(history)}",
        fontsize=13, fontweight='bold'
    )

    for line, (i, j) in zip(edges_lines, G.edges()):
        line.set_data([current_X[i, 0], current_X[j, 0]],
                      [current_X[i, 1], current_X[j, 1]])

    return [scatter] + edges_lines

ani = animation.FuncAnimation(
    fig, update, frames=len(history), init_func=init,
    interval=30, blit=False  
)

ani.save("charitha_formation.mp4", writer=animation.FFMpegWriter(fps=30))
