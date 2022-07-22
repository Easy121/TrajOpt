import numpy as np


class SquareGrid:
    def __init__(self, x_range, y_range, step, obs):
        self.nodes = set()
        self.node2edges = dict()
        
        self.x_range = x_range
        self.y_range = y_range
        self.step = step
        self.diagstep = 1.414213562373 * self.step
        self.obs = obs
        
        for x in np.arange(self.x_range[0], self.x_range[1], self.step):
            for y in np.arange(self.y_range[0], self.y_range[1], self.step):
                node = (x, y)
                if node not in self.obs:
                    self.nodes.add(node)
                    neighbors = [
                        (x+self.step, y), 
                        (x-self.step, y), 
                        (x, y-self.step), 
                        (x, y+self.step), 
                    ]  # E W N S
                    neighbors_good = []
                    for neighbor in neighbors:
                        if self.isBounded(neighbor) and self.isPassable(neighbor):
                            neighbors_good.append(neighbor)
                    self.node2edges[node] = neighbors_good
    
    def getCost(self, current, next):
        return self.step
    
    def isBounded(self, id):
        (x, y) = id
        return self.x_range[0] <= x < self.x_range[1] and self.y_range[0] <= y < self.y_range[1]
    
    def isPassable(self, id):
        return id not in self.obs
    
    def neighbors(self, id):
        return self.node2edges[id]


class SquareGridDiagonal:
    def __init__(self, x_range, y_range, step, obs):
        self.nodes = set()
        self.node2edges = dict()
        
        self.x_range = x_range
        self.y_range = y_range
        self.step = step
        self.diagstep = 1.414213562373 * self.step
        self.obs = obs
        
        for x in np.arange(self.x_range[0], self.x_range[1], self.step):
            for y in np.arange(self.y_range[0], self.y_range[1], self.step):
                node = (x, y)
                if node not in self.obs:
                    self.nodes.add(node)
                    neighbors = [
                        (x+self.step, y), 
                        (x-self.step, y), 
                        (x, y-self.step), 
                        (x, y+self.step), 
                        (x+self.step, y+self.step), 
                        (x-self.step, y-self.step), 
                        (x-self.step, y+self.step), 
                        (x+self.step, y-self.step), 
                    ]  # E W N S and diagonal
                    neighbors_good = []
                    for neighbor in neighbors:
                        if self.isBounded(neighbor) and self.isPassable(neighbor):
                            neighbors_good.append(neighbor)
                    self.node2edges[node] = neighbors_good
    
    def getCost(self, current, next):
        if next[0] == current[0] or next[1] == current[1]:
            return self.step
        else:
            return self.diagstep
    
    def isBounded(self, id):
        (x, y) = id
        return self.x_range[0] <= x < self.x_range[1] and self.y_range[0] <= y < self.y_range[1]
    
    def isPassable(self, id):
        return id not in self.obs
    
    def neighbors(self, id):
        return self.node2edges[id]