"""
A test for double lane change (DLC) track reference generation
Use splines / Bezier and straights to form the reference
"""


from opt.calc import *

import os
import numpy as np
from scipy.interpolate import BPoly
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


""" User inputs """
w = 1.22  # maximum vehicle width


""" Track Specification """
start_1 = 0
end_1 = 12
start_2 = end_1
end_2 = start_2 + 13.5
start_3 = end_2
end_3 = start_3 + 11
start_4 = end_3
end_4 = start_4 + 12.5
start_5 = end_4
end_5 = start_5 + 12

# extension
start_6 = end_5
# end_6 = start_6 + 12
end_6 = start_6 + 20

bl_1 = (1.1*w+0.25)/2
br_1 = -(1.1*w+0.25)/2
bl_3 = (1.1*w+0.25)/2+1+(w+1)
br_3 = (1.1*w+0.25)/2+1
bl_5 = np.maximum((1.3*w+0.25), 3)/2
br_5 = -np.maximum((1.3*w+0.25), 3)/2


""" Bezier """
s = 4  # length parameter
n = 400  # number of points per section
X = np.linspace(0, 1, n, endpoint=False)  # sample index from 0 to 1

# 1
sampled_points_1 = np.vstack((np.linspace(start_1, end_1, n,endpoint=False), np.array([0]*n))).T
# 3
sampled_points_3 = np.vstack((np.linspace(start_3, end_3, n,endpoint=False), np.array([(bl_3+br_3)/2]*n))).T
# 5
sampled_points_5 = np.vstack((np.linspace(start_5, end_5, n,endpoint=False), np.array([0]*n))).T
# 6
sampled_points_6 = np.vstack((np.linspace(start_6, end_6, n,endpoint=False), np.array([0]*n))).T

# points in section 2
contorl_points_2 = np.array([[start_2, 0], 
                             [start_2+s, 0], [start_2+s, 0], 
                             [end_2-s, (bl_3+br_3)/2], [end_2-s, (bl_3+br_3)/2], 
                             [end_2, (bl_3+br_3)/2]])
curve_2 = BPoly(contorl_points_2[:, np.newaxis, :], [0, 1])
sampled_points_2 = curve_2(X)

# points in section 4
contorl_points_4 = np.array([[start_4, (bl_3+br_3)/2], 
                             [start_4+s, (bl_3+br_3)/2], [start_4+s, (bl_3+br_3)/2], 
                             [end_4-s, 0], [end_4-s, 0], 
                             [end_4, 0]])
curve_4 = BPoly(contorl_points_4[:, np.newaxis, :], [0, 1])
sampled_points_4 = curve_4(X)

# together
sampled_points = np.concatenate((sampled_points_1, 
                                 sampled_points_2, 
                                 sampled_points_3, 
                                 sampled_points_4, 
                                 sampled_points_5,
                                 sampled_points_6))  # extend the track



""" Results """
calc_final = Calc(sampled_points)
length_final = calc_final.Length()
kappa_final = calc_final.CurvO1F(open=True)
kappa_final = np.append(kappa_final, [0, 0])  # ensure same dimension
theta_final = calc_final.ThetaAtan(open=True)
theta_final = np.append(theta_final, [0])  # ensure same dimension
dis_final = calc_final.Dis()
# fake boundaries
Bl = np.array([0]*length_final.size)
Br = np.array([0]*length_final.size)
print('Number of points: ', length_final.size)



""" Velocity Profile Generation """
# ay = vx^2 / R = vx^2 * kappa, vx_max = sqrt(ay_max/kappa)
# vx_max = 10
# ay_max = 5
vx_max = 17
ay_max = 10
vx = []
for i in range(kappa_final.size):
    if kappa_final[i] != 0:
        vx.append(np.minimum(vx_max, np.sqrt(np.abs(ay_max/kappa_final[i]))))
    else:
        vx.append(vx_max)
vx = np.asarray(vx)


""" Plot """
# plot the lane change boundary
# 1
ax[0].plot([start_1, end_1], [bl_1]*2, '-', color=CL['BLK'], label='Boundary', linewidth=3)
ax[0].plot([start_1, end_1], [br_1]*2, '-', color=CL['BLK'], linewidth=3)
# 3
ax[0].plot([start_3, end_3], [bl_3]*2, '-', color=CL['BLK'], linewidth=3)
ax[0].plot([start_3, end_3], [br_3]*2, '-', color=CL['BLK'], linewidth=3)
# 5
ax[0].plot([start_5, end_5], [bl_5]*2, '-', color=CL['BLK'], linewidth=3)
ax[0].plot([start_5, end_5], [br_5]*2, '-', color=CL['BLK'], linewidth=3)
# reference
ax[0].plot(sampled_points[0::5,0], sampled_points[0::5,1], '--', color=CL['BLU'], label='Reference', linewidth=3)

# other properties
ax[1].plot(length_final, kappa_final, '-', color=CL['BLU'],
            label='Reference curvature', linewidth=3, markersize=8)
ax[2].plot(length_final, theta_final, '-', color=CL['BLU'],
            label='Reference $\\theta$ (yaw of track)', linewidth=3, markersize=8)
# ax[3].plot(dis_final, '-', color=CL['BLU'],
#             label='Distance', linewidth=3, markersize=8)
ax[3].plot(length_final, vx, '-', color=CL['BLU'],
            label='Velocity Profile', linewidth=3, markersize=8)

# ax[0].axis('equal')
ax[1].set_xlim([length_final[0], length_final[-1]])
ax[2].set_xlim([length_final[0], length_final[-1]])
ax[3].set_xlim([length_final[0], length_final[-1]])

ax[0].set_xlabel('X ($m$)', fontsize=15)
ax[0].set_ylabel('Y ($m$)', fontsize=15)
ax[1].set_xlabel('Curve length ($m$)', fontsize=15)
ax[1].set_ylabel('Curvature', fontsize=15)
ax[2].set_xlabel('Curve length ($m$)', fontsize=15)
ax[2].set_ylabel('$\\theta$ ($rad$)', fontsize=15)
# ax[3].set_xlabel('Curve length ($m$)', fontsize=15)
# ax[3].set_ylabel('Distance ($m$)', fontsize=15)
ax[3].set_xlabel('Curve length ($m$)', fontsize=15)
ax[3].set_ylabel('$v_x$ ($m/s$)', fontsize=15)
ax[0].legend(loc='upper right', fontsize=10)
ax[1].legend(loc='lower right', fontsize=10)
ax[2].legend(loc='upper right', fontsize=10)
ax[3].legend(loc='lower right', fontsize=10)
# ax[4].legend(loc='lower right', fontsize=10)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
ax[2].grid(linestyle='--')
ax[3].grid(linestyle='--')
# ax[4].grid(linestyle='--')
plt.tight_layout()
plt.show()