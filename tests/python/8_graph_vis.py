import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
CL = {'BLU': np.array([0, 114, 189])/255,
      'LBLU': np.array([173, 216, 230])/255,
      'RED': np.array([217, 83, 25])/255,
      'LRED': np.array([255, 204, 203])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


# graph generation
d = 1
G = nx.grid_2d_graph(np.arange(0, 15, d), np.arange(0, 15, d))

# vis
print(G.nodes)
print(type(G.nodes))
# print(np.asarray(G.nodes))
print(len(G))
print(list(G).index((1, 2)))

plt.ion()
xy = np.asarray(G.nodes)
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=80)
ax.scatter(xy[:,0], xy[:, 1], s=30**2, marker='s')
ax.axis('equal')
plt.tight_layout()
plt.show()

