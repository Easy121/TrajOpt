"""
Comparing curvature optimization Fixed and Free 
"""


from opt.opt import *
from opt.calc import *

import numpy as np
import yaml
import os
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],})
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


""" Data """
interval = 0.5
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'map.txt')
left_cones = []
right_cones = []
with open(path_to_track) as f:
    for line in f.readlines():
        temp = []
        # split according to space
        l = line.split(' ')
        for i in l[:-1]:
            temp.append(float(i))
        if(l[-1] == '1\n'):
            left_cones.append(temp)
        if(l[-1] == '2\n'):
            right_cones.append(temp)
left_cones = np.asarray(left_cones).reshape(-1, 2)
right_cones = np.asarray(right_cones).reshape(-1, 2)
path_to_right = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'reference_right_sparse.yaml')
path_to_center = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'reference_center_sparse.yaml')
with open(path_to_right) as stream:
    right = yaml.safe_load(stream)
with open(path_to_center) as stream:
    center = yaml.safe_load(stream)
right_points = np.hstack((np.asarray(right['x']).reshape(-1,1), np.asarray(right['y']).reshape(-1,1)))
center_points = np.hstack((np.asarray(center['x']).reshape(-1,1), np.asarray(center['y']).reshape(-1,1)))
right_calc = Calc(right_points)
center_calc = Calc(center_points)


""" Plot """
# 1. path comparison
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
ax.plot(left_cones[:, 0], left_cones[:, 1], '.', color=CL['RED'], label='Left cones', linewidth=2, markersize=8)
ax.plot(right_cones[:, 0], right_cones[:, 1], '.', color=CL['BLU'], label='Right cones', linewidth=2, markersize=8, zorder=10)
ax.plot(right['x'], right['y'], '-', color=CL['ORA'], label='CO-Fixed', linewidth=3, markersize=8)
ax.plot(center['x'], center['y'], '-', color=CL['BLU'], label='CO-Free', linewidth=3, markersize=8)
ax.set_xlabel('X ($m$)', fontsize=20)
ax.set_ylabel('Y ($m$)', fontsize=20)
ax.axis('equal')
ax.legend(loc='upper right', fontsize=15)  # 标签位置
ax.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "1 Path comparison of CO-Fixed and CO-Free.svg"))

# 2. curvature comparison
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
ax.plot(right_calc.Length() / right_calc.Length()[-1], right_calc.CurvO1F(), '-', color=CL['ORA'], label='CO-Fixed', linewidth=3, markersize=8)
ax.plot(center_calc.Length() / center_calc.Length()[-1], center_calc.CurvO1F(), '-', color=CL['BLU'], label='CO-Free', linewidth=3, markersize=8)
ax.set_xlabel('Curve length ($\%$)', fontsize=20)
ax.set_ylabel('Curvature', fontsize=20)
ax.set_xlim([0, 1])
ax.legend(loc='upper right', fontsize=15)  # 标签位置
ax.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "2 Curvature comparison of CO-Fixed and CO-Free.svg"))

# 3. boundary distance
fig, ax = plt.subplots(figsize=(10, 6), dpi=80)
ax.plot(right_calc.Length() / right_calc.Length()[-1], right['bl'], '-', color=CL['ORA'], label='CO-Fixed', linewidth=3, markersize=8)
ax.plot(right_calc.Length() / right_calc.Length()[-1], right['br'], '-', color=CL['ORA'], linewidth=3, markersize=8)
ax.plot(center_calc.Length() / center_calc.Length()[-1], center['bl'], '-', color=CL['BLU'], label='CO-Free', linewidth=3, markersize=8)
ax.plot(center_calc.Length() / center_calc.Length()[-1], center['br'], '-', color=CL['BLU'], linewidth=3, markersize=8)
ax.set_xlabel('Curve length ($\%$)', fontsize=20)
ax.set_ylabel('Boundary distance ($m$)', fontsize=20)
ax.set_xlim([0, 1])
ax.legend(loc='upper right', fontsize=15)  # 标签位置
ax.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "3 Boundary distance comparison of CO-Fixed and CO-Free.svg"))

# 4. (subfigure) Distance error analysis of CO-Fixed
s_error_right     = right_calc.SError(interval)  # n-1 data
s_error_max_index_right = np.argmax(s_error_right)
s_error_center     = center_calc.SError(interval)  # n-1 data

fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
ax.plot(right_points[s_error_max_index_right, 0], right_points[s_error_max_index_right, 1], '.', color=CL['BLK'], label='Largest error', markersize=14, zorder=9)
ax.plot(left_cones[:, 0], left_cones[:, 1], '.', color=CL['RED'], label='Left cones', linewidth=2, markersize=8)
ax.plot(right_cones[:, 0], right_cones[:, 1], '.', color=CL['BLU'], label='Right cones', linewidth=2, markersize=8, zorder=10)
ax.plot(right['x'], right['y'], '-', color=CL['ORA'], label='CO-Fixed', linewidth=3, markersize=8)
ax.set_xlabel('X ($m$)', fontsize=20)
ax.set_ylabel('Y ($m$)', fontsize=20)
ax.legend(loc='upper right', fontsize=15)  # 标签位置
ax.axis('equal')
ax.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "4_1 Point with largest error.svg"))

fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
ax.plot(right_calc.Length()[:-1] / right_calc.Length()[-1], s_error_right/interval*100, '-', color=CL['ORA'], label='CO-Fixed', linewidth=3, markersize=8)
ax.plot(center_calc.Length()[:-1] / center_calc.Length()[-1], s_error_center/interval*100, '-', color=CL['BLU'], label='CO-Free', linewidth=3, markersize=8)
ax.set_xlabel('Curve length ($\%$)', fontsize=20)
ax.set_ylabel('Percent $s_{error}$ ($\%$)', fontsize=20)
ax.set_xlim([0, 1])
ax.legend(loc='upper left', fontsize=15)  # 标签位置
ax.grid(linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), "4_2 Distance error comparison.svg"))

# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.xaxis.set_ticks([])
# ax.yaxis.set_ticks([])

# plt.pause(0.01)  # 暂停
plt.show()