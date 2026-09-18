import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# PART 1: N = 5 Simulation
N1 = 5
p1 = 0.5
G1 = nx.erdos_renyi_graph(N1, p1, seed=42)

while not nx.is_connected(G1):
    G1 = nx.erdos_renyi_graph(N1, p1)

A1 = nx.adjacency_matrix(G1).toarray()
D1 = np.diag(np.sum(A1, axis=1))
M1 = nx.incidence_matrix(G1, oriented=True).toarray()
L1 = nx.laplacian_matrix(G1).toarray()

eigenvalues1 = np.sort(np.linalg.eigvalsh(L1))
lambda2_1 = eigenvalues1[1]

print("--- PART 1 RESULTS ---")
print("Adjacency Matrix:\n", A1)
print("Degree Matrix:\n", D1)
print("Incidence Matrix:\n", M1)
print("Laplacian Matrix:\n", L1)
print("Laplacian Eigenvalues:", np.round(eigenvalues1, 4))
print(f"Algebraic Connectivity (lambda_2): {lambda2_1:.4f}")

plt.figure(figsize=(5, 4))
pos1 = nx.spring_layout(G1, seed=42)
nx.draw(G1, pos1, with_labels=True, node_color='skyblue', node_size=700, font_weight='bold')
plt.title(f"Erdos-Renyi Graph (N=5), lambda_2={lambda2_1:.3f}")
plt.savefig("er_n5.png")
plt.close()


# PART 2: N = 1000 Simulation
N2 = 1000
p2 = 0.015
G2 = nx.erdos_renyi_graph(N2, p2, seed=100)

# Rejection-sampling to guarantee connectivity
while not nx.is_connected(G2):
    G2 = nx.erdos_renyi_graph(N2, p2)

degrees2 = dict(G2.degree())
max_deg = max(degrees2.values())
max_deg_nodes = [node for node, deg in degrees2.items() if deg == max_deg]

mean_degree_emp = np.mean(list(degrees2.values()))
mean_degree_theo = p2 * (N2 - 1)
mean_edges_emp = G2.number_of_edges()
mean_edges_theo = p2 * N2 * (N2 - 1) / 2

print("\n=============== PART 2 RESULTS =====================")
print(f"Theoretical Mean Degree: {mean_degree_theo:.3f} | Empirical: {mean_degree_emp:.3f}")
print(f"Theoretical Mean Edges: {mean_edges_theo:.1f} | Empirical: {mean_edges_emp}")

diameter_val = nx.diameter(G2)
print(f"Graph Diameter: {diameter_val}")

pairs_dict = dict(nx.all_pairs_shortest_path_length(G2))
diam_source, diam_target = None, None
for u in pairs_dict:
    for v in pairs_dict[u]:
        if pairs_dict[u][v] == diameter_val:
            diam_source, diam_target = u, v
            break
    if diam_source is not None:
        break

diameter_path = nx.shortest_path(G2, source=diam_source, target=diam_target)
print(f"Diameter path between node {diam_source} and {diam_target}: {diameter_path}")

# 2. sp for a random pair of nodes
source, target = np.random.choice(G2.nodes(), size=2, replace=False)
sp_path = nx.shortest_path(G2, source=source, target=target, weight=None)
print(f"Shortest path between node {source} and {target}: {sp_path}")

plt.figure(figsize=(12, 10))
pos2 = nx.spring_layout(G2, k=0.15, seed=42)

node_colors = ['magenta' if n in max_deg_nodes else 'lightgray' for n in G2.nodes()]
node_sizes = [80 if n in max_deg_nodes else 15 for n in G2.nodes()]

diam_edges = list(zip(diameter_path[:-1], diameter_path[1:]))
sp_edges = list(zip(sp_path[:-1], sp_path[1:]))

edge_colors = []
edge_widths = []

for e in G2.edges():
    if e in diam_edges or (e[1], e[0]) in diam_edges:
        edge_colors.append('blue')
        edge_widths.append(3.0)
    elif e in sp_edges or (e[1], e[0]) in sp_edges:
        edge_colors.append('red')
        edge_widths.append(2.5)
    else:
        edge_colors.append('gainsboro')
        edge_widths.append(0.2)

nx.draw_networkx_nodes(G2, pos2, node_color=node_colors, node_size=node_sizes)
nx.draw_networkx_edges(G2, pos2, edge_color=edge_colors, width=edge_widths)

plt.title(
    f"Erdos-Renyi Graph (N = 1000)\n"
    f"Magenta: Max Degree Node(s) | Blue: Diameter Path (Length {diameter_val}) | "
    f"Red: Shortest Path ({source} to {target})",
    fontsize=11, pad=20
)
plt.axis('off')
plt.tight_layout()
plt.savefig("er_n1000.png", dpi=300, bbox_inches='tight')
plt.close()
