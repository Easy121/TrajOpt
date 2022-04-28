"""
Generate the boundary npz
"""


from opt.opt import *
from opt.calc import *

import os
import yaml
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"], })
fig, ax = plt.subplots(1, 5, figsize=(25, 5), dpi=80)
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255, }


""" Construct Problem """
# Import data
path_to_data = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'data/map.txt')
left = []
right = []
# map.txt constains coordinate of left and right cones
with open(path_to_data) as f:
    for line in f.readlines():
        temp = []
        # split according to space
        l = line.split(' ')
        for i in l[:-1]:
            temp.append(float(i))
        if(l[-1] == '1\n'):
            left.append(temp)
        if(l[-1] == '2\n'):
            right.append(temp)
left = np.asarray(left).reshape(-1, 2)
right = np.asarray(right).reshape(-1, 2)

interval = 0.05
boundleft = Referenceline_Opt(left,
                              type='opt',
                              interval=interval,
                              a=1.0,
                              b=0.0,
                              g=2.0)
boundright = Referenceline_Opt(right,
                               type='opt',
                               interval=interval,
                               a=1.0,
                               b=0.0,
                               g=2.0)


""" Optimize """
boundleft.optimize()
boundright.optimize()


""" Export """
path_left = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/bound_left.npz')
path_right = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/bound_right.npz')
np.savez(path_left, P=boundleft.P_all_sol)
np.savez(path_right, P=boundright.P_all_sol)
