""" 
Figures of 7dof, run on IPC
The optimization took such a long time. Therefore, the visualization is better based on saved data
"""


from opt.vis import *

import os
import numpy as np
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

""" Data loading """
# * remember to change
# 1. 0.3m interval
path_to_reference = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/reference_7dof_1.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/track.yaml')
path_to_config = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../4_7dof/config.yaml')
path_to_param = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../4_7dof/param.yaml')
# * remember to change
# 1. unconstarined omega
# 2. constarined omega (good)
# path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), '7dof_1.npz')
path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), '7dof_2.npz')

data = np.load(path_to_data)
# Import reference
with open(path_to_reference, 'r') as stream:
    ref = yaml.safe_load(stream)
    
with open(path_to_config) as stream:
    solver_config = yaml.safe_load(stream)
mu = solver_config['mu']
mu_c = solver_config['mu_c']

# Dx and Dy for friction ellipse
with open(path_to_param) as stream:
    vehicle_param = yaml.safe_load(stream)
Dx = np.abs(vehicle_param['wheel']['Dx'])
Dy = np.abs(vehicle_param['wheel']['Dy'])

""" Vis """
# give a full plot
plotter = Plotter(None, None, 12, width_ratios=[0, 0], figsize=(24, 12), mode='debug', interval=1)
plotter.t_sequence = data['t_sequence']
plotter.x_arch = data['x_arch']
plotter.u_arch = data['u_arch']
plotter.dx_arch = data['dx_arch']
plotter.carinfo_arch = data['carinfo_arch']

""" state plot """
plotter.plotTrack(0, path_to_track)
plotter.plotReference(0, path_to_reference, color=CL['BLU'], interval=1)
plotter.plotOptXY(0, 7, ref, '-', legend_loc='upper right')
plotter.plotActualState(1, 2, '-', xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend_loc='lower right')
plotter.plotActualState(2, 1, '-', xlabel='Time ($s$)', ylabel='$v_y$ ($m/s$)', legend_loc='lower right')
plotter.plotActualState(4, 0, '-', xlabel='Time ($s$)', ylabel='$v_x$ ($m/s$)', legend_loc='lower right')
plotter.plotActualState(5, 7, '-', xlabel='Time ($s$)', ylabel='$n$ ($m$)', legend_loc='upper left')

plotter.plotActualState(8, 3, '-', color=CL['RED'], xlabel='Time ($s$)', ylabel='$\omega$ ($rad/s$)', legend='$\omega_{fl}$', legend_loc='lower right')
plotter.plotActualState(8, 4, '-', color=CL['RED']*0.7, xlabel='Time ($s$)', ylabel='$\omega$ ($rad/s$)', legend='$\omega_{fr}$', legend_loc='lower right')
plotter.plotActualState(8, 5, '-', color=CL['BLU'], xlabel='Time ($s$)', ylabel='$\omega$ ($rad/s$)', legend='$\omega_{rl}$', legend_loc='lower right')
plotter.plotActualState(8, 6, '-', color=CL['BLU']*0.7, xlabel='Time ($s$)', ylabel='$\omega$ ($rad/s$)', legend='$\omega_{rr}$', legend_loc='lower right')

""" input plot """
plotter.plotActualState(3, 9, '-', xlabel='Time ($s$)', ylabel='$\delta$ ($rad$)', color=CL['ORA'], legend_loc='upper left')
plotter.plotActualState(7, 10, '-', xlabel='Time ($s$)', ylabel='$T_l$ ($Nm$)', color=CL['ORA'], legend_loc='upper right')
# plotter.plotActualInput(11, 0, '-', xlabel='Time ($s$)', ylabel='$\dot{\delta}$ ($rad/s$)', legend_loc='upper left')

""" dstate plot """
plotter.plotActualdState(9, 0, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['BLK'], legend='$\dot{v}_x$', legend_loc='upper right')
plotter.plotActualdState(9, 1, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['ORA'], legend='$\dot{v}_y$', legend_loc='upper right')
plotter.plotActualAx(9, 0, 1, 2, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['RED'], legend='$a_x$', legend_loc='upper right')
plotter.plotActualAy(9, 1, 0, 2, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['BLU'], legend='$a_y$', legend_loc='upper right')
# Vertical Tire force
plotter.plotActualCarInfo(10, 0, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['RED'], legend='$F_{z,fl}$', legend_loc='upper right')
plotter.plotActualCarInfo(10, 1, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['RED']*0.7, legend='$F_{z,fr}$', legend_loc='upper right')
plotter.plotActualCarInfo(10, 2, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['BLU'], legend='$F_{z,rl}$', legend_loc='upper right')
plotter.plotActualCarInfo(10, 3, '-', omit_start=1, xlabel='Time ($s$)', ylabel='$F_z$ ($N$)', color=CL['BLU']*0.7, legend='$F_{z,rf}$', legend_loc='upper right')
# Long. and Lat. Tire force
plotter.plotFrictionEllipse(11, [Dx, Dy], mu, legend='Fric. Ell.')
plotter.plotFrictionEllipse(11, [Dx, Dy], mu_c, legend='Constr. Ell.')
plotter.plotActualTireForceDless(11, 4, 8, 0, '-', color=CL['RED'], legend='$F_{fl}$', legend_loc='upper right', markeredgewidth=0.5, interval=20)
plotter.plotActualTireForceDless(11, 5, 9, 1, '-', color=CL['RED']*0.7, legend='$F_{fr}$', legend_loc='upper right', markeredgewidth=0.5, interval=20)
plotter.plotActualTireForceDless(11, 6, 10, 2, '-', color=CL['BLU'], legend='$F_{rl}$', legend_loc='upper right', markeredgewidth=0.5, interval=20)
plotter.plotActualTireForceDless(11, 7, 11, 3, '-', color=CL['BLU']*0.7, legend='$F_{rr}$', legend_loc='upper right', markeredgewidth=0.5, interval=20)
plotter.plotActualTireForceDlessEnd(11, 4, 8, 0, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')
plotter.plotActualTireForceDlessEnd(11, 5, 9, 1, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')
plotter.plotActualTireForceDlessEnd(11, 6, 10, 2, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')
plotter.plotActualTireForceDlessEnd(11, 7, 11, 3, '+', zorder=5, color=CL['WHT'], legend_loc='upper right')

plt.show()
