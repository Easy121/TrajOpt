""" 
Generate reference path from trajectory optimization for vehicle tracking
"""


from opt.calc import *
from opt.opt import *
from opt.opt_traj import *
from opt.vis import *

import os
import yaml
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
fig, ax = plt.subplots(1, 5, figsize=(25, 5), dpi=80)
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


""" Path """

path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../7dof_2_closed.npz')
# path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../7dof_noa_closed.npz')
path_to_reference = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/reference_7dof_1.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/track.yaml')
path_to_bound_left = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/bound_left.npz')
path_to_bound_right = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/bound_right.npz')
path_to_map = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/map.txt')


""" Data """

# data
data = np.load(path_to_data)
x_arch = data['x_arch'][1:,:]
vx = x_arch[:,0]
# Import reference
with open(path_to_reference, 'r') as stream:
    ref = yaml.safe_load(stream)
# Import track
with open(path_to_track, 'r') as stream:
    track = yaml.safe_load(stream)
# calc XY
# x_ref = np.append(np.asarray(ref["x"]), ref["x"][0])
# y_ref = np.append(np.asarray(ref["y"]), ref["y"][0])
# theta_ref = np.append(np.asarray(ref["theta"]), ref["theta"][0])
x_ref = np.asarray(ref["x"])
y_ref = np.asarray(ref["y"])
theta_ref = np.asarray(ref["theta"])
n_arch = x_arch[:,7]
X = x_ref - n_arch * np.sin(theta_ref)
Y = y_ref + n_arch * np.cos(theta_ref)
P = np.vstack((X, Y)).T
# left right boundary
left = []
right = []
# map.txt constains coordinate of left and right cones
with open(path_to_map) as f:
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
# optimized left right boundary
bound_left = np.load(path_to_bound_left)
bound_right = np.load(path_to_bound_right)


""" Process """
interval = 0.05


""" Results """
# # Final curve length, curvature and yaw angle of track -> theta
calc_final = Calc(P)
length_final = calc_final.Length()
s_error     = calc_final.SError(interval)  # n-1 data
s_error_max_index = np.argmax(s_error)
kappa_final = calc_final.CurvO1F()
theta_final = calc_final.ThetaAtan()
# Final left and right boundary distance and absolute position
Bl, _, Bl_pos, _ = calc_final.BoundExternRight(bound_left['P'])
_ , Br, _, Br_pos = calc_final.BoundExternLeft(bound_right['P'])
# # Absolute position of boundary points
# print('Number of fixed points: ', ref.n_fixed)
# print('Number of free points : ', ref.n_free)
# The difference between theta start and end should be about 2*pi
print('Theta start              : ', np.rad2deg(theta_final[0]), 'deg')
print('Theta end                : ', np.rad2deg(theta_final[-1]), 'deg')
print('Theta start - (end + 360): ',
      np.rad2deg(theta_final[0])-(np.rad2deg(theta_final[-1])+360), 'deg')
print('Total length: ', length_final[-1])


""" Plot """
# # plot intersections of closest points
# for index_left_based in indexes_left_based:
#     x = left[index_left_based[0], 0], right[index_left_based[1], 0]
#     y = left[index_left_based[0], 1], right[index_left_based[1], 1]
#     ax[0].plot(x, y, '--', color=CL['RED'], linewidth=1, markersize=8)
# for index_right_based in indexes_right_based:
#     x = left[index_right_based[0], 0], right[index_right_based[1], 0]
#     y = left[index_right_based[0], 1], right[index_right_based[1], 1]
#     ax[0].plot(x, y, ':', color=CL['BLU'], linewidth=1, markersize=8)

# ax[0].plot(np.append(P_fixed[:, 0], P_fixed[0, 0]),
#             np.append(P_fixed[:, 1], P_fixed[0, 1]), '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1, markersize=8)
ax[0].plot(P[:, 0], P[:, 1], '.',
           color=CL['BLU'], label='Optimized reference', linewidth=3, markersize=4)
ax[0].plot(left[:, 0], left[:, 1], '.', color=CL['RED'],
            label='Left cones', linewidth=2, markersize=8)
ax[0].plot(right[:, 0], right[:, 1], '.', color=CL['BLU'],
            label='Right cones', linewidth=2, markersize=8)
ax[0].plot(Bl_pos[:, 0], Bl_pos[:, 1], '-', color=CL['GRE'],
            label='Left boundary', linewidth=1, markersize=6)
ax[0].plot(Br_pos[:, 0], Br_pos[:, 1], '-', color=CL['GRE'], 
            label='Right boundary', linewidth=1, markersize=6)
ax[1].plot(length_final, kappa_final, '-', color=CL['BLU'],
            label='Optimized curvature', linewidth=3, markersize=8)
ax[2].plot(length_final, theta_final, '-', color=CL['BLU'],
            label='Optimized $\\theta$ (yaw of track)', linewidth=3, markersize=8)
ax[3].plot(length_final, Bl, '-', color=CL['RED'],
            label='Left boundary distance', linewidth=3, markersize=8)
ax[3].plot(length_final, Br, '-', color=CL['BLU'],
            label='Right boundary distance', linewidth=3, markersize=8)
ax[4].plot(length_final[0:-1], s_error/interval*100, '-', color=CL['BLU'],
            label='Road progress error', linewidth=3, markersize=8)
ax[0].axis('equal')
ax[1].set_xlim([length_final[0], length_final[-1]])
ax[2].set_xlim([length_final[0], length_final[-1]])
ax[3].set_xlim([length_final[0], length_final[-1]])
# # ax[1].set_ylim([-100, 100])
ax[0].set_xlabel('X ($m$)', fontsize=15)
ax[0].set_ylabel('Y ($m$)', fontsize=15)
ax[1].set_xlabel('Curve length ($m$)', fontsize=15)
ax[1].set_ylabel('Curvature', fontsize=15)
ax[2].set_xlabel('Curve length ($m$)', fontsize=15)
ax[2].set_ylabel('$\\theta$ ($rad$)', fontsize=15)
ax[3].set_xlabel('Curve length ($m$)', fontsize=15)
ax[3].set_ylabel('Boundary distance ($m$)', fontsize=15)
ax[4].set_xlabel('Curve length ($m$)', fontsize=15)
ax[4].set_ylabel('Percent $s_{error}$ (%)', fontsize=15)
ax[0].legend(loc='lower right', fontsize=10)
ax[1].legend(loc='lower right', fontsize=10)
ax[2].legend(loc='upper right', fontsize=10)
ax[3].legend(loc='lower right', fontsize=10)
ax[4].legend(loc='lower right', fontsize=10)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
ax[2].grid(linestyle='--')
ax[3].grid(linestyle='--')
ax[4].grid(linestyle='--')
plt.tight_layout()
plt.show()
