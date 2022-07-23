from re import A
import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(np.vstack((x, y)).T)

a = np.vstack((np.linspace(0, 5, 4,endpoint=False), np.array([0]*4))).T
print(a)
print(np.concatenate((a, a, a)))


""" List and ndarray """
print('')
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
print(x/y)
# y = np.array([4, 5])
# print(x/y)
x = np.array([1, 2, 3, 4, 5])
print(np.concatenate((x[:3], [1, 2], x[3:])).tolist())


print('')
init = [-16, -17, 0.5]
ref = [-19, -20, 0.6]
print(np.max([np.abs(init[0]), np.abs(ref[0])]))
print(np.max([np.abs(init[1]), np.abs(ref[1])]))


print('')
print(list(np.array([1, 2, 3]) / np.array([2, 4, 6])))
print(type(list(np.array([1, 2, 3]) / np.array([2, 4, 6]))))


print('')
print(np.square(1))
print(np.min([1, 2]))


print('')
x = [1]
x = x + [1, 2]
print(x)

print('')
# print(np.linalg.norm((2, 2)-(1, 2)))
x = [(1, 2), (2, 2) ,(3, 2)]
i = 0
y = {x_element:i for x_element in x}

print('')
x = []
x.append([1, (1, 2)])
print(x)

print('')
d = 0.1
all_nodes = []
for x in np.arange(0, 1, d):
    all_nodes.append((x, x))
print(all_nodes)
print(type(all_nodes[1][0]))
test = (d*3, d*3)
print(type(test[0]))
print(test == all_nodes[3])

print('')
y = np.array([1, 2])
print(np.append(np.array([0, 0]), y))

print('')
print(0 % 360)

print('')
for i in range(20):
    if i == 5:
        break
print(i)