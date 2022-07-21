import heapq


pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task


task = (1, 2)
priority = 1

entry = [priority, task]
entry_finder[task] = entry
heapq.heappush(pq, entry)

print(pq)

entry = entry_finder.pop(task)
entry[-1] = REMOVED

print(pq)


