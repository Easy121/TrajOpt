""" 
Test of greedy algorithm
* don't consider the traveled distance from start
* only consider the remaining distance to end
"""


import numpy as np
import networkx as nx
import heapq
from typing import Protocol, Dict, List, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
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


class PriorityQueue:
    def __init__(self):
        self.elements: List[Tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]


# graph generation
d = 1
G = nx.grid_2d_graph(np.arange(0, 15, d), np.arange(0, 15, d))
# setting obstacles
# TODO automatic setting
obs = [(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2),
       (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12),
       (11, 12), (10, 12), (9, 12), (8, 12), (7, 12), (6, 12), (5, 12)]
G.remove_nodes_from(obs)
# start and end point
start = (0, 2)
goal  = (11, 13)
# search initialization
frontier = PriorityQueue()
frontier.put(start, 0)
came_from = dict()  # path A->B is stored as came_from[B] == A
cost_so_far = dict()
came_from[start] = None
cost_so_far[start] = 0
# heuristic: Euclidean
def heuristic(a, b):
   return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
# search loop
while not frontier.empty():
    current = frontier.get()
    # early exit
    if current == goal:
        break
    # expand the frontier
    for next in G.neighbors(current):
        new_cost = cost_so_far[current]  # no uniform cost here (from start)
        if next not in cost_so_far or new_cost < cost_so_far[next]:
            cost_so_far[next] = new_cost
            priority = new_cost + heuristic(goal, next)  # greedy term (to end)
            frontier.put(next, priority)
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
