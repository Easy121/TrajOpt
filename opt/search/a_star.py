from . import queue


import numpy as np


def aStarCost(start_state, start, goal, G):
    # heuristic: Euclidean
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)*2
    
    def euclidean(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    # search initialization
    frontier = queue.PriorityQueue()
    frontier.put(start, 0)
    came_from = dict()  # path A->B is stored as came_from[B] == A
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        # early exit
        if current == goal:
            break
        
        # expand the frontier
        for next in G.neighbors(current):
            new_cost = cost_so_far[current] + G.getCost(current, next)  # uniform cost (from start)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)  # greedy term (to end) 
                # the priority term should not be accumulated as next_cost does, therefore placed here
                frontier.put(next, priority)
                came_from[next] = current
                
    # backward
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    if len(path) > 4:
        # discrete cost
        cost_d = cost_so_far[goal] - cost_so_far[path[1]]
        # continuous cost
        cost_c = euclidean(start_state, path[1])
        return cost_d + cost_c
    else:
        return euclidean(start_state, goal)
