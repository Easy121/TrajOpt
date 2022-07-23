""" 
Test of Hybrid A* algorithm
* consider multiple heuristic
"""
# TODO Visualization of continuous points
# TODO Redd-sheep curve heuristic


from opt.stats import *
profprinter = ProfPrinter()
from opt.search.queue import *
from opt.search.grid import *
from opt.search.a_star import *


import numpy as np
import cProfile as profile
import os
import itertools
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
# TODO automatic setting
d = 1
obs = {(2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (12, 2),
       (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12),
       (11, 12), (10, 12), (9, 12), (8, 12), (7, 12), (6, 12), (5, 12)}
# G = SquareGrid([0, 15], [0, 15], 1, obs)
G = SquareGridDiagonal([0, 15], [0, 15], d, obs)

# start and end point
start = (0, 2)
goal  = (11, 13)
# goal  = (13, 7)
# continuous state
state = {start: [0.0, 2.0, 90]}

# visual 
color_change_all = []
G_set = G.nodes
G_list = list(G_set)
xy = np.array(G_list)
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
# load map
path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'map_Psi2increm.npy')
Psi2increm = np.load(path_to_save, allow_pickle='TRUE').item()

# heuristic: Euclidean
def heuristic(a, b):
   return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)*2

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
    current_test = frontier.get()
    if current_test is None:
        break
    
    current = current_test
    current_state = state[current]
    color_change = [[node2index[current], CL['LRED']]]
    
    # early exit
    if current == goal:
        break
    
    # expand the frontier
    for increm in Psi2increm[current_state[2] % 360]:
        next_state = [current_state[0] + increm[0], 
                      current_state[1] + increm[1],
                      current_state[2] + increm[2]]
        found = False
        for next in itertools.chain(G.neighbors(current), [current]):
            if abs(next[0] - next_state[0]) < d / 2.0 and abs(next[1] - next_state[1]) < d / 2.0:
                found = True
                break
        if found:
            if next != current:
                new_cost = cost_so_far[current] + d  # uniform cost (from start)
                if next not in cost_so_far or new_cost <= cost_so_far[next]:
                    priority = new_cost + aStarCost(next_state, next, goal, G)  # greedy term (to end) 
                    if frontier.put(next, priority):
                        cost_so_far[next] = new_cost
                        state[next] = next_state
                        came_from[next] = current
                        color_change.append([node2index[next], CL['RED']])
            else:
                new_cost = cost_so_far[current]  # uniform cost (from start)
                priority = new_cost + aStarCost(next_state, next, goal, G)  # greedy term (to end) 
                if frontier.put(next, priority):
                    cost_so_far[next] = new_cost
                    state[next] = next_state
                    # came_from[next] = current  # if not commented, the bug of self came from self arises
                    color_change.append([node2index[next], CL['RED']])
            
    # vis
    color_change_all.append(color_change)

# backward
path = []
path_state = []
while current != start:
    path.append(current)
    path_state.append(state[current])
    current = came_from[current]
path.append(start)
path_state.append(state[start])
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

# init
im = []
path_state = np.asarray(path_state)

# final plot
color_change_final = []
for path_point in path:
    color_change_final.append([node2index[path_point], CL['BLU']])

# aggregate
for color_change in color_change_all:
    for one_change in color_change:
        node_color[one_change[0]] = one_change[1]
    im.append([ax.scatter(xy[:,0], xy[:, 1], s=30**2, c=node_color, marker='s')])
# final
for one_change in color_change_final:
    node_color[one_change[0]] = one_change[1]
# together with line
for i in range(30):
    im.append([ax.scatter(xy[:,0], xy[:, 1], s=30**2, c=node_color, marker='s'),
               ax.plot(path_state[:, 0], path_state[:, 1], '.-', color=CL['BLK'], linewidth=3, markersize=16)[0]])

# clear plot and reset setting
ax.cla()
plt.tight_layout()
ax.axis('equal')

# animation
ani = animation.ArtistAnimation(fig, im, interval=50, repeat=True, blit=True)

# save: comment ax.cla() to save
# path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'fig/Hybrid A*.gif')
# ani.save(filename=path_to_save, writer='pillow')

###################################################
prof.disable() ####################################
###################################################
profprinter.print(prof)

plt.show()

