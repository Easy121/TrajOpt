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
    os.path.dirname(__file__))), 'referenceline/data/reference_center_sparse.yaml')
    # os.path.dirname(__file__))), 'referenceline/data/reference_right_sparse.yaml')
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
plotter.plotOptState(7, 4, '-', xlabel='Time ($s$)', ylabel='$\chi$ ($rad$)', legend_loc='upper left')

""" input plot """
plotter.plotOptState(2, 5, '-', xlabel='Time ($s$)', color=CL['ORA'], ylabel='$\delta$ ($rad$)', legend_loc='upper right')
plotter.plotOptInput(5, 1, '-', xlabel='Time ($s$)', ylabel='$T_l$ ($Nm$)', legend_loc='upper right')

vis_t = time.perf_counter() - vis_t0
print("[TIME] Visualization takes: %.3f s" % vis_t) # CPU seconds elapsed (floating point)
plt.show()
