""" 
Copyright (C) 2021：
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


import os
import sys
path_current = os.path.abspath(os.path.dirname(__file__))
path_bag = os.path.dirname(os.path.dirname(path_current))
sys.path.insert(0, path_bag) 
from opt.opt import *
from opt.calc import *

import csv
import yaml
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"], })
fig, ax = plt.subplots(1, 4, figsize=(20, 5), dpi=80)
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
    os.path.dirname(__file__)), 'test.txt')
left = []
right = []
P_fixed = []
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

P_fixed_Left = Calc.CentersLeft(left, right)
P_fixed_Min, indexes_left_based, indexes_right_based = Calc.CentersMin(left, right)
P_fixed_Vote, linkage = Calc.CentersVote(left, right)
print(linkage)

""" Plotting """
# plot intersections of closest points
for index_left_based in indexes_left_based:
    x = left[index_left_based[0], 0], right[index_left_based[1], 0]
    y = left[index_left_based[0], 1], right[index_left_based[1], 1]
    ax[1].plot(x, y, '--', color=CL['RED'], linewidth=1, markersize=8)
for index_right_based in indexes_right_based:
    x = left[index_right_based[0], 0], right[index_right_based[1], 0]
    y = left[index_right_based[0], 1], right[index_right_based[1], 1]
    ax[1].plot(x, y, ':', color=CL['BLU'], linewidth=1, markersize=8)

ax[0].plot(P_fixed_Left[:, 0], P_fixed_Left[:, 1], '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1, markersize=8)
ax[1].plot(P_fixed_Min[:, 0], P_fixed_Min[:, 1], '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1, markersize=8)
ax[2].plot(P_fixed_Vote[:, 0], P_fixed_Vote[:, 1], '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1, markersize=8)
# ax.plot(traj.P_sol[:, 0], traj.P_sol[:, 1], '-',
#         color=CL['BLU'], label='Optimized waypoints', linewidth=3)
ax[0].plot(left[:, 0], left[:, 1], '.-', color=CL['RED'],
        label='Left cones', linewidth=3, markersize=8)
ax[0].plot(right[:, 0], right[:, 1], '.-', color=CL['BLU'],
        label='Right cones', linewidth=3, markersize=8)
ax[1].plot(left[:, 0], left[:, 1], '.-', color=CL['RED'],
        label='Left cones', linewidth=3, markersize=8)
ax[1].plot(right[:, 0], right[:, 1], '.-', color=CL['BLU'],
        label='Right cones', linewidth=3, markersize=8)
ax[2].plot(left[:, 0], left[:, 1], '.-', color=CL['RED'],
        label='Left cones', linewidth=3, markersize=8)
ax[2].plot(right[:, 0], right[:, 1], '.-', color=CL['BLU'],
        label='Right cones', linewidth=3, markersize=8)
ax[0].axis('equal')
ax[1].axis('equal')
ax[2].axis('equal')
# ax[1].set_xlim([length_final[0], length_final[-1]])
ax[0].set_xlabel('X ($m$)', fontsize=15)
ax[0].set_ylabel('Y ($m$)', fontsize=15)
ax[1].set_xlabel('X ($m$)', fontsize=15)
ax[1].set_ylabel('Y ($m$)', fontsize=15)
ax[2].set_xlabel('X ($m$)', fontsize=15)
ax[2].set_ylabel('Y ($m$)', fontsize=15)
ax[0].legend(loc='upper left', fontsize=10)
ax[1].legend(loc='upper left', fontsize=10)
ax[2].legend(loc='upper left', fontsize=10)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
ax[2].grid(linestyle='--')
plt.tight_layout()
plt.show()