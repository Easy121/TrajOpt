import numpy as np
import casadi as ca


a = ca.SX.sym('a')
b = ca.SX.sym('b')
c = ca.SX.sym('c')

print('')
x = np.array([1, 2, 3])
print(b - x * a)

print('')
y = ca.vertcat(a, b, c)
print(y)
print(y[0])
print((b - x * a) * y)

print('')
z = ca.vertcat(a, b, 0.0, 0.0)
print(z)

print('')
B1 = ca.SX.zeros(3)
print(B1)
B1[0] = a
B1[1] = b
B1[2] = c
print(B1)
print(ca.sum1(B1))
print(ca.vertcat((b - x * a) * y, a))
