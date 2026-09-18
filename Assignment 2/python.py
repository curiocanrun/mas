import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

G = nx.Graph()
edges = [
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 5),
    (2, 6),
    (2, 8),
    (2, 4),
    (3, 5),
    (3, 7),
    (3, 8),
    (3, 4),
    (5, 6),
    (6, 8),
    (8, 7),
    (7, 5),
    (5, 8),
    (6, 7),
    (8, 4),
]
G.add_edges_from(edges)

N, F, T = 8, 1, 25
np.random.seed(42)

x = np.zeros((N, T))
x[:7, 0] = np.random.uniform(10, 50, 7)

# node 8 is malicious
mal = 7

# malicious node's state
for k in range(T):
  x[mal, k] = 80 + 20 * np.sin(k)

# W-MSR updates
for k in range(T - 1):

  x_next = x[:, k].copy()

  for i in range(N):

    # malicious node does not follow W-MSR
    if i == mal:
      continue

    # Neighbors + node itself
    nbrs = [j - 1 for j in G.neighbors(i + 1)] + [i]

    # store (node, value)
    values = [(j, x[j, k]) for j in nbrs]

    xi = x[i, k]

    # values strictly smaller than x_i (sorted ascending to discard smallest)
    lower = sorted([(j, v) for j, v in values if v < xi], key=lambda p: p[1])

    # values strictly larger than x_i (sorted descending to discard largest)
    upper = sorted(
        [(j, v) for j, v in values if v > xi], key=lambda p: p[1], reverse=True
    )

    # discard up to F values on each side of x_i
    discard = set()

    for j, _ in lower[:F]:
      discard.add(j)

    for j, _ in upper[:F]:
      discard.add(j)

    # remaining values
    kept = [v for j, v in values if j not in discard]

    if len(kept) == 0:
      kept = [xi]

    x_next[i] = np.mean(kept)

  # store all benign updates at once
  x[:, k + 1] = x_next

  # keep malicious node's predefined value
  x[mal, k + 1] = 80 + 20 * np.sin(k + 1)


plt.figure(figsize=(8, 4.5))

for i in range(N):
  ls = 'r--' if i == mal else '-'
  plt.plot(
      x[i, :], ls, label=f'Node {i+1}' + (' (Mal)' if i == mal else '')
  )

plt.xlabel('Time step $k$')
plt.ylabel('State $x_i[k]$')
plt.title('W-MSR resilient consensus ($F=1$)')
plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('wmsr_.png')
plt.close()
