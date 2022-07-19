import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
CL = {'BLU': np.array([0, 114, 189])/255,
      'LBLU': np.array([173, 216, 230])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


# graph generation
d = 1
G = nx.grid_2d_graph(np.arange(0, 10, d), np.arange(0, 10, d))
# setting obstacles
obs = [(4, 4), (5, 4), (4, 5), (5, 5)]
G.remove_nodes_from(obs)



""" Inspecting """
# print
# print(G.nodes)
# draw
pos = {(x,y):(x,y) for x,y in G.nodes()}  # tuple (x, y) unpacked
node_color = [CL['LBLU'] for x,y in G.nodes()]  # tuple (x, y) unpacked
nx.draw(G, pos=pos, 
        node_color=node_color, 
        # with_labels=True,
        )
plt.axis('equal')
plt.show()
