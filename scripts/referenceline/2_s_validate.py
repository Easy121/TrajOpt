"""
A test for reference line optimization
It compares the prescribed distance with the real value, and visualize the percent error
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
# Define opt class
ref = Referenceline_Opt(right,
                        type='opt',
                        interval=interval,
                        a=1.0,
                        b=0.0,
                        g=2.0)


""" Optimize """
ref.optimize()


""" Results """
# # Final curve length, curvature and yaw angle of track -> theta
calc_final = Calc(ref.P_all_sol)
length_final = calc_final.Length()
s_error     = calc_final.SError(interval)  # n-1 data
s_error_max_index = np.argmax(s_error)
kappa_final = calc_final.CurvO1F()
theta_final = calc_final.ThetaAtan()
# Absolute position of boundary points
print('Number of fixed points: ', ref.n_fixed)
print('Number of free points : ', ref.n_free)
# The difference between theta start and end should be about 2*pi
print('Theta start              : ', np.rad2deg(theta_final[0]), 'deg')
print('Theta end                : ', np.rad2deg(theta_final[-1]), 'deg')
print('Theta start - (end + 360): ',
      np.rad2deg(theta_final[0])-(np.rad2deg(theta_final[-1])+360), 'deg')
print('Total length: ', length_final[-1])


""" Export """
# format data
# output = {
#     'x': ref.P_all_sol[:, 0].tolist(),
#     'y': ref.P_all_sol[:, 1].tolist(),
#     's': length_final.tolist(),
#     'kappa': kappa_final.tolist(),
#     'theta': theta_final.tolist(),
#     'bl': Bl.tolist(),
#     'br': Br.tolist(),
# }
# path_to_output = os.path.join(os.path.abspath(
#     os.path.dirname(__file__)), 'data/' + 'reference' + '.yaml')
# with open(path_to_output, 'w') as stream:
#     yaml.dump(output, stream)


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
ax[0].plot(ref.P_all_sol[:, 0], ref.P_all_sol[:, 1], '.',
           color=CL['BLU'], label='Optimized reference', linewidth=3, markersize=6)
# high light the maximum error
ax[0].plot(ref.P_all_sol[s_error_max_index, 0], ref.P_all_sol[s_error_max_index, 1], '.',
           color=CL['BLK'], label='Largest error', markersize=14)
ax[0].plot(left[:, 0], left[:, 1], '.', color=CL['RED'],
            label='Left cones', linewidth=2, markersize=12)
ax[0].plot(right[:, 0], right[:, 1], '.', color=CL['BLU'],
            label='Right cones', linewidth=2, markersize=12)
# ax[0].plot(Bl_pos[:, 0], Bl_pos[:, 1], '-', color=CL['GRE'],
#             label='Left boundary', linewidth=2, markersize=6)
# ax[0].plot(Br_pos[:, 0], Br_pos[:, 1], '.', color=CL['GRE'], 
            # linewidth=2, markersize=6)
ax[1].plot(length_final, kappa_final, '-', color=CL['BLU'],
            label='Optimized curvature', linewidth=3, markersize=8)
ax[2].plot(length_final, theta_final, '-', color=CL['BLU'],
            label='Optimized $\\theta$ (yaw of track)', linewidth=3, markersize=8)
ax[3].plot(length_final[0:-1], s_error/interval*100, '-', color=CL['BLU'],
            label='Road progress error', linewidth=3, markersize=8)
# ax[3].plot(length_final, Bl, '-', color=CL['RED'],
#             label='Left boundary distance', linewidth=3, markersize=8)
# ax[3].plot(length_final, Br, '-', color=CL['BLU'],
#             label='Right boundary distance', linewidth=3, markersize=8)
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
ax[3].set_ylabel('Percent $s_{error}$ (%)', fontsize=15)
ax[0].legend(loc='lower right', fontsize=10)
ax[1].legend(loc='lower right', fontsize=10)
ax[2].legend(loc='upper right', fontsize=10)
ax[3].legend(loc='upper left', fontsize=10)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
ax[2].grid(linestyle='--')
ax[3].grid(linestyle='--')
plt.tight_layout()
plt.show()
