"""
A test for centerline optimization
It treats the centerline, which is constructed through complex calculation from the cone locations,
as reference line.
"""


from opt.opt import *
from opt.calc import *

import os
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
# Import settings
path_to_config = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'config.yaml')
with open(path_to_config, 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
# Import data
path_to_data = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'data/map.txt')
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
_, indexes_left_based, indexes_right_based = Calc.CentersMin(left, right)
P_fixed, linkage = Calc.CentersVote(left, right)
# Define opt class
if config['opt_type'] == 'centerline':
    traj = Referenceline_Opt(P_fixed,
                             type=config['centerline_option']['type'],
                             interval=config['centerline_option']['interval'],
                             a=config['centerline_option']['a'],
                             b=config['centerline_option']['b'],
                             g=config['centerline_option']['g'])


""" Optimize """
traj.optimize()


""" Results """
# Final curve length, curvature and yaw angle of track -> theta
calc_final = Calc(traj.P_all_sol, config['vehicle'])
length_final = calc_final.Length()
kappa_final = calc_final.CurvO1F()
theta_final = calc_final.ThetaAtan()
# Final left and right boundary distance and absolute position
Bl_final, Br_final, Bl_pos_final, Br_pos_final = calc_final.BoundLinear(left, right)
# Absolute position of boundary points
print('Number of fixed points: ', traj.n_fixed)
print('Number of free points : ', traj.n_free)
# The difference between theta start and end should be about 2*pi
print('Theta start              : ', np.rad2deg(theta_final[0]), 'deg')
print('Theta end                : ', np.rad2deg(theta_final[-1]), 'deg')
print('Theta start - (end + 360): ',
      np.rad2deg(theta_final[0])-(np.rad2deg(theta_final[-1])+360), 'deg')
print('Total length: ', length_final[-1])


""" Export """
# format data
output = {
    'x': traj.P_all_sol[:, 0].tolist(),
    'y': traj.P_all_sol[:, 1].tolist(),
    'length': length_final.tolist(),
    'kappa': kappa_final.tolist(),
    'theta': theta_final.tolist(),
}
path_to_output = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'data/' + config['opt_type'] + '_' + config['centerline_option']['type'] + '.yaml')
with open(path_to_output, 'w') as stream:
    yaml.dump(output, stream)


""" Plot """
if config['enable_plot'] == True:
    # plot intersections of closest points
    for index_left_based in indexes_left_based:
        x = left[index_left_based[0], 0], right[index_left_based[1], 0]
        y = left[index_left_based[0], 1], right[index_left_based[1], 1]
        ax[0].plot(x, y, '--', color=CL['RED'], linewidth=1, markersize=8)
    for index_right_based in indexes_right_based:
        x = left[index_right_based[0], 0], right[index_right_based[1], 0]
        y = left[index_right_based[0], 1], right[index_right_based[1], 1]
        ax[0].plot(x, y, ':', color=CL['BLU'], linewidth=1, markersize=8)

    ax[0].plot(np.append(P_fixed[:, 0], P_fixed[0, 0]),
               np.append(P_fixed[:, 1], P_fixed[0, 1]), '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1, markersize=8)
    ax[0].plot(traj.P_sol[:, 0], traj.P_sol[:, 1], '-',
               color=CL['BLU'], label='Optimized waypoints', linewidth=3)
    ax[0].plot(left[:, 0], left[:, 1], '.', color=CL['RED'],
               label='Left cones', linewidth=3, markersize=8)
    ax[0].plot(right[:, 0], right[:, 1], '.', color=CL['BLU'],
               label='Right cones', linewidth=3, markersize=8)
    ax[0].plot(Bl_pos_final[:, 0], Bl_pos_final[:, 1], '-', color=CL['GRE'],
               label='Shrinked boundaries', linewidth=2, markersize=8)
    ax[0].plot(Br_pos_final[:, 0], Br_pos_final[:, 1], '-', color=CL['GRE'], 
               linewidth=2, markersize=8)
    ax[1].plot(length_final, kappa_final, '-', color=CL['BLU'],
               label='Optimized curvature', linewidth=3, markersize=8)
    ax[2].plot(length_final, theta_final, '-', color=CL['BLU'],
               label='Optimized $\\theta$ (yaw of track)', linewidth=3, markersize=8)
    ax[3].plot(length_final, Bl_final, '-', color=CL['RED'],
               label='Left boundary distance (shrinked)', linewidth=3, markersize=8)
    ax[3].plot(length_final, Br_final, '-', color=CL['BLU'],
               label='Right boundary distance (shrinked)', linewidth=3, markersize=8)
    ax[0].axis('equal')
    ax[1].set_xlim([length_final[0], length_final[-1]])
    ax[2].set_xlim([length_final[0], length_final[-1]])
    ax[3].set_xlim([length_final[0], length_final[-1]])
    # ax[1].set_ylim([-100, 100])
    ax[0].set_xlabel('X ($m$)', fontsize=15)
    ax[0].set_ylabel('Y ($m$)', fontsize=15)
    ax[1].set_xlabel('Curve length ($m$)', fontsize=15)
    ax[1].set_ylabel('Curvature', fontsize=15)
    ax[2].set_xlabel('Curve length ($m$)', fontsize=15)
    ax[2].set_ylabel('$\\theta$ ($rad$)', fontsize=15)
    ax[3].set_xlabel('Curve length ($m$)', fontsize=15)
    ax[3].set_ylabel('Boundary distance ($m$)', fontsize=15)
    ax[0].legend(loc='lower right', fontsize=10)
    ax[1].legend(loc='lower right', fontsize=10)
    ax[2].legend(loc='upper right', fontsize=10)
    ax[3].legend(loc='upper right', fontsize=10)
    ax[0].grid(linestyle='--')
    ax[1].grid(linestyle='--')
    ax[2].grid(linestyle='--')
    ax[3].grid(linestyle='--')
    plt.tight_layout()
    plt.show()
