import heapq
from typing import TypeVar
T = TypeVar('T')


REMOVED = (-10000, -10000)  # placeholder for a removed task


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.sheet = dict()
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        if item not in self.sheet:
            entry = [priority, item]
            self.sheet[item] = entry
            heapq.heappush(self.elements, entry)
        elif self.sheet[item][0] > priority:
            # lazy deletiion of the former one
            former = self.sheet.pop(item)
            former[-1] = REMOVED
            entry = [priority, item]
            self.sheet[item] = entry
            heapq.heappush(self.elements, entry)
    
    def get(self) -> T:
        # remove the lazy terms
        while self.elements:
            item_poped = heapq.heappop(self.elements)[1]
            if item_poped is not REMOVED:
                del self.sheet[item_poped]
                return item_poped
        raise KeyError('pop from an empty priority queue')
