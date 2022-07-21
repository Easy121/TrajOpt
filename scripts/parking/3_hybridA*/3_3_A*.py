""" 
Test of greedy algorithm
* don't consider the traveled distance from start
* only consider the remaining distance to end
"""
# DONE Animation
# TODO non-repeating priority queue


import numpy as np
import networkx as nx
import time
import cProfile as profile
import pstats
import os
import heapq
from typing import Protocol, Dict, List, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')
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

# visual 
im = []
G_list = list(G)
# creat node index dictionary
index = range(len(G_list))
node2index = dict(zip(G_list, index))
node_color = [CL['LBLU']] * len(G_list)
node_color[node2index[start]] = CL['LRED']
# fig and coordinate
xy = np.array(G)
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=80)
plt.tight_layout()
ax.axis('equal')

# search initialization
frontier = PriorityQueue()
frontier.put(start, 0)
came_from = dict()  # path A->B is stored as came_from[B] == A
cost_so_far = dict()
came_from[start] = None
cost_so_far[start] = 0

# heuristic: Euclidean
def heuristic(a, b):
   return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)*2

# search loop
prof = profile.Profile()
prof.enable()

while not frontier.empty():
    current = frontier.get()
    node_color[node2index[current]] = CL['LRED']
    
    # early exit
    if current == goal:
        break
    
    # expand the frontier
    for next in G.neighbors(current):
        new_cost = cost_so_far[current] + d  # no uniform cost here (from start)
        if next not in cost_so_far or new_cost < cost_so_far[next]:
            cost_so_far[next] = new_cost
            priority = new_cost + heuristic(goal, next)  # greedy term (to end)
            frontier.put(next, priority)
            came_from[next] = current
    
    # # plot
    # for priority_point, frontier_point in frontier.elements:
    #     node_color[node2index[frontier_point]] = CL['RED']
    # im.append([ax.scatter(xy[:,0], xy[:, 1], s=30**2, c=node_color, marker='s')])

prof.disable()

# backward
current = goal
path = []
while current != start:
    path.append(current)
    current = came_from[current]
path.append(start)
path.reverse()

# final plot
for path_point in path:
    node_color[node2index[path_point]] = CL['BLU']
for i in range(20):
    im.append([ax.scatter(xy[:,0], xy[:, 1], s=30**2, c=node_color, marker='s')])


""" Inspecting """
# print
# print(G.nodes)

# print profiling output
stats = pstats.Stats(prof).strip_dirs().sort_stats('cumtime')
# stats.print_stats(10) # top 10 rows

def f8(x):
    return "%8.6f" % x

def func_std_string(func_name): # match what old profile produced
    if func_name[:2] == ('~', 0):
        # special case for built-in functions
        name = func_name[2]
        if name.startswith('<') and name.endswith('>'):
            return '{%s}' % name[1:-1]
        else:
            return name
    else:
        return "%s:%d(%s)" % func_name

print('')
indent = ' ' * 8
print(indent, stats.total_calls, "function calls", end=' ')
if stats.total_calls != stats.prim_calls:
    print("(%d primitive calls)" % stats.prim_calls, end=' ')
print("in %.6f seconds" % stats.total_tt)
print('')
width, list = stats.get_print_list([10])
if list:
    print('   ncalls  tottime  percall  cumtime  percall', end=' ')
    print('filename:lineno(function)')
    for func in list:
        cc, nc, tt, ct, callers = stats.stats[func]
        c = str(nc)
        if nc != cc:
            c = c + '/' + str(cc)
        print(c.rjust(9), end=' ')
        print(f8(tt), end=' ')
        if nc == 0:
            print(' '*8, end=' ')
        else:
            print(f8(tt/nc), end=' ')
        print(f8(ct), end=' ')
        if cc == 0:
            print(' '*8, end=' ')
        else:
            print(f8(ct/cc), end=' ')
        print(func_std_string(func))
print('')


""" Plotting """
# ani = animation.ArtistAnimation(fig, im, interval=50, repeat=True, blit=True)
# path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'A*.gif')
# ani.save(filename=path_to_save, writer='pillow')
# plt.show()

# * discarded
# # position
# pos = {(x,y):(x,y) for x,y in G.nodes()}  # tuple (x, y) unpacked
# # color
# node_color = []
# for x, y in G.nodes:
#     if (x,y) in path:
#         node_color.append(CL['BLU'])
#     elif (x,y) in came_from:
#         node_color.append(CL['LRED'])
#     else:
#         node_color.append(CL['LBLU'])
# # plot
# nx.draw(G, pos=pos, 
#         node_color=node_color,
#         # node_size=20
#         # with_labels=True,
#         )
# plt.axis('equal')
# plt.show()


