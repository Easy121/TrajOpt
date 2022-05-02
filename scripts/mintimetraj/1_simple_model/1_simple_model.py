"""
A test for minimum time optimization
Simple 3dof model
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
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


""" Path """

path_to_reference = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/reference_center_sparse.yaml')
# path_to_reference = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/reference_right_sparse.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/track.yaml')
path_to_config = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'config.yaml')
path_to_param = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'param.yaml')


""" Data """

# mu for Vis
with open(path_to_config) as stream:
    solver_config = yaml.safe_load(stream)
mu = solver_config['mu']
mu_c = solver_config['mu_c']
# Dx and Dy for friction ellipse
with open(path_to_param) as stream:
    vehicle_param = yaml.safe_load(stream)
Dx = np.abs(vehicle_param['wheel']['Dx'])
Dy = np.abs(vehicle_param['wheel']['Dy'])


""" Construct Problem """

# Import reference
with open(path_to_reference, 'r') as stream:
    ref = yaml.safe_load(stream)
# Import track
with open(path_to_track, 'r') as stream:
    track = yaml.safe_load(stream)
# Import config
with open(path_to_config, 'r') as stream:
    config = yaml.safe_load(stream)
# object
traj = Trajectory_Opt(ref, config, 'bi_3dof_ddelta')
# traj = Trajectory_Opt(ref, config, 'bi_3dof_ddelta', closed=True)


""" Optimize """

traj.optimize()


""" Plot """
# * vis.py is the replicate of visual.py from GMPCPlatform, this is an efficient workaround, please keep updated

vis_t0 = time.perf_counter()

# parse data
plotter = Plotter(None, None, 12, width_ratios=[0, 0], figsize=(24, 12), mode='debug', interval=1)
plotter.t_sequence = traj.t_opt
plotter.x_arch = traj.x_opt
plotter.u_arch = traj.u_opt
plotter.dx_arch = traj.dx_opt
plotter.carinfo_arch = traj.carinfo_opt

""" state plot """
plotter.plotTrack(0, path_to_track)
plotter.plotReference(0, path_to_reference, color=CL['BLU'], interval=1)
plotter.plotOptXY(0, 3, ref, '-', legend_loc='upper right')
plotter.plotActualState(1, 2, '-', xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend_loc='lower right')
plotter.plotActualState(2, 1, '-', xlabel='Time ($s$)', ylabel='$v_y$ ($m/s$)', legend_loc='lower right')
plotter.plotActualState(4, 0, '-', xlabel='Time ($s$)', ylabel='$v_x$ ($m/s$)', legend_loc='lower right')
plotter.plotActualState(5, 3, '-', xlabel='Time ($s$)', ylabel='$n$ ($m$)', legend_loc='upper left')

""" input plot """
plotter.plotActualState(3, 5, '-', xlabel='Time ($s$)', ylabel='$\delta$ ($rad$)', color=CL['ORA'], legend_loc='upper left')
plotter.plotActualInput(7, 1, '-', xlabel='Time ($s$)', ylabel='$T_l$ ($Nm$)', legend_loc='upper right')
plotter.plotActualInput(11, 0, '-', xlabel='Time ($s$)', ylabel='$\dot{\delta}$ ($rad/s$)', legend_loc='upper left')

""" dstate plot """
plotter.plotActualdState(8, 0, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['BLK'], legend='$\dot{v}_x$', legend_loc='upper right')
plotter.plotActualdState(8, 1, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['ORA'], legend='$\dot{v}_y$', legend_loc='upper right')
plotter.plotActualAx(8, 0, 1, 2, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['RED'], legend='$a_x$', legend_loc='upper right')
plotter.plotActualAy(8, 1, 0, 2, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['BLU'], legend='$a_y$', legend_loc='upper right')
# Vertical Tire force
plotter.plotActualCarInfo(9, 0, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['RED'], legend='$F_{z,f}$', legend_loc='upper right')
plotter.plotActualCarInfo(9, 1, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['BLU'], legend='$F_{z,r}$', legend_loc='upper right')
# Long. and Lat. Tire force
plotter.plotFrictionEllipse(10, [Dx, Dy], mu, legend='Fric. Ell.')
plotter.plotFrictionEllipse(10, [Dx, Dy], mu_c, legend='Constr. Ell.')
plotter.plotActualTireForceDless(10, 2, 4, 0, '-', omit_start=1, color=CL['RED'], legend='$F_{f}$', legend_loc='upper right', markeredgewidth=0.5, interval=1)
plotter.plotActualTireForceDless(10, 3, 5, 1, '-', omit_start=1, color=CL['BLU'], legend='$F_{r}$', legend_loc='upper right', markeredgewidth=0.5, interval=1)
plotter.plotActualTireForceDlessEnd(10, 2, 4, 0, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')
plotter.plotActualTireForceDlessEnd(10, 3, 5, 1, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')

vis_t = time.perf_counter() - vis_t0
print("[TIME] Visualization takes: %.3f s" % vis_t) # CPU seconds elapsed (floating point)
plt.show()

# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/road_reference/center.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/road_reference/right.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/mal_definition/mal_defined.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/mal_definition/correct.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/four_models/3-DOF wo LT.npz'))
