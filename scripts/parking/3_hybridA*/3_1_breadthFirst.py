""" 
Test of breathfirst algorithm
"""

import numpy as np
import networkx as nx
import queue
import matplotlib.pyplot as plt
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
# setting obstacles
obs = [(4, 4), (5, 4), (4, 5), (5, 5)]
G.remove_nodes_from(obs)
# start and end point
start = (0, 0)
goal  = (5, 8)
# search initialization
frontier = queue.Queue()
frontier.put(start)
came_from = dict()  # path A->B is stored as came_from[B] == A
came_from[start] = None
# search loop
while not frontier.empty():
    current = frontier.get()
    # early exit
    if current == goal:
        break
    # expand the frontier
    for next in G.neighbors(current):
        if next not in came_from:
            frontier.put(next)
            came_from[next] = current
# backward
current = goal
path = []
while current != start:
    path.append(current)
    current = came_from[current]
path.append(start)
path.reverse()


""" Inspecting """
# print
# print(G.nodes)


""" Plotting """
# position
pos = {(x,y):(x,y) for x,y in G.nodes()}  # tuple (x, y) unpacked
# color
node_color = []
for x, y in G.nodes:
    if (x,y) in path:
        node_color.append(CL['BLU'])
    elif (x,y) in came_from:
        node_color.append(CL['LRED'])
    else:
        node_color.append(CL['LBLU'])
# plot
nx.draw(G, pos=pos, 
        node_color=node_color,
        # node_size=20
        # with_labels=True,
        )
plt.axis('equal')
plt.show()
