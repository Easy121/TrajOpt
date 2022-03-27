"""
A test for minimum time optimization
It the simple 3dof model for equality constraints
"""


from opt.calc import *
from opt.opt import *
from opt.opt_traj import *
from opt.vis import *

import os
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,}

""" Construct Problem """
# Import reference
path_to_reference = os.path.join(os.path.abspath(os.path.dirname(
    os.path.dirname(__file__))), 'referenceline/data/reference_sparse.yaml')
with open(path_to_reference, 'r') as stream:
    ref = yaml.safe_load(stream)
# Import track
path_to_track = os.path.join(os.path.abspath(os.path.dirname(
    os.path.dirname(__file__))), 'referenceline/data/track.yaml')
with open(path_to_track, 'r') as stream:
    track = yaml.safe_load(stream)

traj = Trajectory_Opt(ref)


""" Optimize """
traj.optimize()


""" Plot """
vis_t0 = time.perf_counter()

plotter = Plotter(ref, track, traj, 12, figsize=(18, 12.5))

""" state plot """
plotter.plotTrack(0)
plotter.plotReference(0)
plotter.plotOptXY(0, 3, '-', legend_loc='upper right')
plotter.plotOptState(1, 2, '-', xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend_loc='lower right')
plotter.plotOptState(3, 0, '-', xlabel='Time ($s$)', ylabel='$v_x$ ($m/s$)', legend_loc='lower right')
plotter.plotOptState(4, 1, '-', xlabel='Time ($s$)', ylabel='$v_y$ ($m/s$)', legend_loc='lower right')
plotter.plotOptState(6, 3, '-', xlabel='Time ($s$)', ylabel='$n$ ($m$)', legend_loc='upper left')

""" input plot """
plotter.plotOptState(2, 5, '-', xlabel='Time ($s$)', color=CL['ORA'], ylabel='$\delta$ ($rad$)', legend_loc='upper right')
plotter.plotOptInput(5, 1, '-', xlabel='Time ($s$)', ylabel='$T_l$ ($Nm$)', legend_loc='upper right')


# ax[0].plot(ref.P_all_sol[:, 0], ref.P_all_sol[:, 1], '-',
#            color=CL['BLU'], label='Optimized reference', linewidth=3)
# ax[0].plot(lx, ly, '.', color=CL['RED'],
#             label='Left cones', linewidth=2, markersize=8)
# ax[0].plot(rx, ry, '.', color=CL['BLU'],
#             label='Right cones', linewidth=2, markersize=8)
# ax[0].plot(Bl_pos[:, 0], Bl_pos[:, 1], '-', color=CL['GRE'],
#             label='Left boundary', linewidth=2, markersize=6)
# ax[0].plot(Br_pos[:, 0], Br_pos[:, 1], '.', color=CL['GRE'], 
            # linewidth=2, markersize=6)
# ax[1].plot(length_final, kappa_final, '-', color=CL['BLU'],
#             label='Optimized curvature', linewidth=3, markersize=8)
# ax[2].plot(length_final, theta_final, '-', color=CL['BLU'],
#             label='Optimized $\\theta$ (yaw of track)', linewidth=3, markersize=8)
# ax[3].plot(length_final, Bl, '-', color=CL['RED'],
#             label='Left boundary distance', linewidth=3, markersize=8)
# ax[3].plot(length_final, Br, '-', color=CL['BLU'],
#             label='Right boundary distance', linewidth=3, markersize=8)
# ax[0].axis('equal')
# ax[1].set_xlim([length_final[0], length_final[-1]])
# ax[2].set_xlim([length_final[0], length_final[-1]])
# ax[3].set_xlim([length_final[0], length_final[-1]])
# # ax[1].set_ylim([-100, 100])
# ax[0].set_xlabel('X ($m$)', fontsize=15)
# ax[0].set_ylabel('Y ($m$)', fontsize=15)
# ax[1].set_xlabel('Curve length ($m$)', fontsize=15)
# ax[1].set_ylabel('Curvature', fontsize=15)
# ax[2].set_xlabel('Curve length ($m$)', fontsize=15)
# ax[2].set_ylabel('$\\theta$ ($rad$)', fontsize=15)
# ax[3].set_xlabel('Curve length ($m$)', fontsize=15)
# ax[3].set_ylabel('Boundary distance ($m$)', fontsize=15)
# ax[0].legend(loc='lower right', fontsize=10)
# ax[1].legend(loc='lower right', fontsize=10)
# ax[2].legend(loc='upper right', fontsize=10)
# ax[3].legend(loc='lower right', fontsize=10)
# ax[0].grid(linestyle='--')
# ax[1].grid(linestyle='--')
# ax[2].grid(linestyle='--')
# ax[3].grid(linestyle='--')
# plt.tight_layout()
vis_t = time.perf_counter() - vis_t0
print("[TIME] Visualization takes: %.3fs" % vis_t) # CPU seconds elapsed (floating point)
plt.show()
# tgrid = np.linspace(0, T, N+1)
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, x_opt[0], '--')
# plt.plot(tgrid, x_opt[1], '-')
# plt.step(tgrid, np.append(np.nan, u_opt[0]), '-.')
# plt.xlabel('t')
# plt.legend(['x1','x2','u'])
# plt.grid()
# plt.show()