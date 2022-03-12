""" 
Copyright (C) 2021:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


from opt.opt import *
from opt.calc import *
import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=80)
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,}


""" Construct Problem """
path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data/demo')
# Real track from [thin]
P_fixed = np.load(os.path.join(path_to_data, 'track_center_data.npy'))
ax[0].plot(np.append(P_fixed[:,0],P_fixed[0,0]),\
        np.append(P_fixed[:,1],P_fixed[0,1]), '.-', color=CL['BLK'], label='Fixed points and connection', linewidth=1)
# Cubic spline data
cubic_spline_dis = np.load(os.path.join(path_to_data, 'cubic_spline_dis.npy')).T
cubic_spline_kappa = np.load(os.path.join(path_to_data, 'cubic_spline_kappa.npy')).T
ax[1].plot(cubic_spline_dis, cubic_spline_kappa, '-', color=CL['BLK'], label='Cubic spline curvature', linewidth=3, markersize=12)
# Define opt class
traj = Referenceline_Opt(P_fixed, 
                         interval=0.05,
                         a=1.0,
                         b=0.0,
                         g=2.0)


""" Optimize """
traj.optimize()


""" Results """
# Final curvature and dis
calc_final = Calc(traj.P_all_sol)
kappa_final = calc_final.CurvO1F()
# kappa_final = calc_final.CurvO2C()
length_final   = calc_final.Length()
print('Number of fixed points: ', traj.n_fixed)
print('Number of free points: ', traj.n_free)
print('Total Length: ', length_final[-1])


""" Plotting final config """
ax[0].scatter(traj.P0[:,0],traj.P0[:,1], label='Initial poisitions', linewidth=1, color=CL['BRO'], marker='.')
# Final poisitions
ax[0].scatter(traj.P_sol[:,0],traj.P_sol[:,1], label='Optimized waypoints', linewidth=1, color=CL['RED'], marker='.')
ax[0].axis('equal')
ax[1].plot(length_final, kappa_final, '-', color=CL['RED'], label='Optimized curvature', linewidth=3, markersize=12)
ax[1].set_xlim([length_final[0], length_final[-1]])
# ax[1].set_ylim([-100, 100])
ax[0].set_xlabel('X ($m$)', fontsize=20)
ax[0].set_ylabel('Y ($m$)', fontsize=20)
ax[1].set_xlabel('Curve length ($m$)', fontsize=20)
ax[1].set_ylabel('Curvature', fontsize=20)
ax[0].legend(loc='lower right', fontsize=15)
ax[1].legend(loc='lower right', fontsize=15)
ax[0].grid(linestyle='--')
ax[1].grid(linestyle='--')
plt.tight_layout()
plt.show()
