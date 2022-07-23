""" 
Test of A* algorithm
* consider heuristic
"""
# DONE Animation
# DONE non-repeating priority queue
# DONE animation simplication
# DONE test heapdict: very slow
# TODO A* diagonal


from opt.stats import *
profprinter = ProfPrinter()
from opt.search.queue import *


import numpy as np
import networkx as nx
import cProfile as profile
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


""" Initialization """
prof = profile.Profile()
###################################################
prof.enable() #####################################
###################################################

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

# visual 
color_change_all = []
G_list = list(G)
xy = np.array(G)
# creat node index dictionary
index = range(len(G_list))
node2index = dict(zip(G_list, index))
node_color = [CL['LBLU']] * len(G_list)
node_color[node2index[start]] = CL['LRED']

# search initialization
frontier = PriorityQueue()
frontier.put(start, 0)
came_from = dict()  # path A->B is stored as came_from[B] == A
cost_so_far = dict()
came_from[start] = None
cost_so_far[start] = 0

# heuristic: Euclidean
def heuristic(a, b):
   return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)*3

###################################################
prof.disable() ####################################
###################################################
profprinter.print(prof, content=True, num=3)


""" Search Loop """
prof = profile.Profile()
###################################################
prof.enable() #####################################
###################################################


while not frontier.empty():
    current = frontier.get()
    color_change = [[node2index[current], CL['LRED']]]
    
    # early exit
    if current == goal:
        break
    
    # expand the frontier
    for next in G.neighbors(current):
        new_cost = cost_so_far[current] + d  # uniform cost (from start)
        if next not in cost_so_far or new_cost < cost_so_far[next]:
            cost_so_far[next] = new_cost
            priority = new_cost + heuristic(goal, next)  # greedy term (to end) 
            # the priority term should not be accumulated as next_cost does, therefore placed here
            frontier.put(next, priority)
            came_from[next] = current
            color_change.append([node2index[next], CL['RED']])
            
    # vis
    color_change_all.append(color_change)

# backward
path = []
while current != start:
    path.append(current)
    current = came_from[current]
path.append(start)
path.reverse()

###################################################
prof.disable() ####################################
###################################################
profprinter.print(prof, content=True, num=8)


""" Plotting """
prof = profile.Profile()
###################################################
prof.enable() #####################################
###################################################

# fig and coordinate
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=80)
plt.tight_layout()
ax.axis('equal')

# init
im = []

# final plot
color_change = []
for path_point in path:
    color_change.append([node2index[path_point], CL['BLU']])
color_change_all += [color_change] * 20

# aggregate
for color_change in color_change_all:
    for one_change in color_change:
        node_color[one_change[0]] = one_change[1]
    im.append([ax.scatter(xy[:,0], xy[:, 1], s=30**2, c=node_color, marker='s')])

# animation
ani = animation.ArtistAnimation(fig, im, interval=50, repeat=True, blit=True)

# save
# path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fig/A*.gif')
# ani.save(filename=path_to_save, writer='pillow')

###################################################
prof.disable() ####################################
###################################################
profprinter.print(prof)

plt.show()

