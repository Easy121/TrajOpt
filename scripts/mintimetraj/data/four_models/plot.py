""" 
Figures of road reference comparison
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
path_to_ref_center = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/reference_center_sparse.yaml')
path_to_ref_7dof_1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/reference_7dof_1.yaml')
path_to_ref_right = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/reference_right_sparse.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/track.yaml')
path_to_3DOFnoLT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '3-DOF wo LT.npz')
path_to_3DOFLT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '3-DOF LT.npz')
path_to_7DOFnoLT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '7-DOF wo LT.npz')
path_to_7DOFLT = os.path.join(os.path.abspath(os.path.dirname(__file__)), '7-DOF LT.npz')

data_3DOFnoLT = np.load(path_to_3DOFnoLT)
data_3DOFLT = np.load(path_to_3DOFLT)
data_7DOFnoLT = np.load(path_to_7DOFnoLT)
data_7DOFLT = np.load(path_to_7DOFLT)
# Import reference
with open(path_to_ref_center, 'r') as stream:
    ref_center = yaml.safe_load(stream)
with open(path_to_ref_right, 'r') as stream:
    ref_right = yaml.safe_load(stream)
with open(path_to_ref_7dof_1, 'r') as stream:
    ref_7dof_1 = yaml.safe_load(stream)

""" Vis """
# plotter = Plotter(None, None, 12, width_ratios=[0, 0], figsize=(24, 12), mode='debug', interval=1)
# plotter.t_sequence = data['t_sequence']
# plotter.x_arch = data['x_arch']
# plotter.u_arch = data['u_arch']
# plotter.dx_arch = data['dx_arch']
# plotter.carinfo_arch = data['carinfo_arch']

# # 1. path
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(10, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotTrack(0, path_to_track)
# plotter.plotReference(0, path_to_ref_center, color=CL['BLK'], interval=1)
# plotter.plotOptXY(0, 3, ref_center, '-', color=CL['RED']*0.7, legend='3-DOF w/o LT', legend_loc='upper right')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotOptXY(0, 3, ref_center, '-', color=CL['RED'], legend='3-DOF LT', legend_loc='upper right')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotOptXY(0, 7, ref_center, '-', color=CL['BLU']*0.7, legend='7-DOF w/o LT', legend_loc='upper right')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotOptXY(0, 7, ref_7dof_1, '-', color=CL['BLU'], legend='7-DOF LT', legend_loc='upper right')

# size = 15
# t = plotter.ax[0].text(
#     50, 0, "$1^{st}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     53, -26, "$2^{nd}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     84, -40, "$3^{rd}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     30, -34, "$4^{th}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     13, -20, "$5^{th}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     0, -50, "$6^{th}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# t = plotter.ax[0].text(
#     -14, 0, "$7^{th}$", ha="center", va="center", rotation=0, size=size,
#     bbox=dict(boxstyle="circle,pad=0.3", facecolor="w", edgecolor="k", alpha=1, lw=2))
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Path comparison of four models.pdf')

# # 2. vx
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 0, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$v_x$ ($m/s$)', legend='3-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 0, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$v_x$ ($m/s$)', legend='3-DOF LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 0, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$v_x$ ($m/s$)', legend='7-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 0, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$v_x$ ($m/s$)', legend='7-DOF LT', legend_loc='lower right')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Longitudinal velocity comparison of four models.pdf')

# # 3. vy
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 1, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$v_y$ ($m/s$)', legend='3-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 1, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$v_y$ ($m/s$)', legend='3-DOF LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 1, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$v_y$ ($m/s$)', legend='7-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 1, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$v_y$ ($m/s$)', legend='7-DOF LT', legend_loc='lower left')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Lateral velocity comparison of four models.pdf')

# # 4. psi
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 2, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='3-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 2, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='3-DOF LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 2, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='7-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 2, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='7-DOF LT', legend_loc='lower left')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Yaw rate comparison of four models.pdf')

# # 5. psi
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 3, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$n$ ($m$)', legend='3-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 3, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$n$ ($m$)', legend='3-DOF LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 7, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$n$ ($m$)', legend='7-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 7, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$n$ ($m$)', legend='7-DOF LT', legend_loc='lower left')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/n comparison of four models.pdf')

# # 6. delta
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
# plotter.t_sequence = data_3DOFnoLT['t_sequence']
# plotter.x_arch = data_3DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 5, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='3-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_3DOFLT['t_sequence']
# plotter.x_arch = data_3DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 5, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='3-DOF LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFnoLT['t_sequence']
# plotter.x_arch = data_7DOFnoLT['x_arch']
# plotter.plotActualStateProgress(0, 9, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='7-DOF w/o LT', legend_loc='upper left')
# plotter.t_sequence = data_7DOFLT['t_sequence']
# plotter.x_arch = data_7DOFLT['x_arch']
# plotter.plotActualStateProgress(0, 9, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='7-DOF LT', legend_loc='lower left')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Steering comparison of four models.pdf')

# 6. delta
plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(6, 5), mode='debug', interval=1)
plotter.t_sequence = data_3DOFnoLT['t_sequence']
plotter.u_arch = data_3DOFnoLT['u_arch']
plotter.plotActualInputProgress(0, 1, '-', color=CL['RED']*0.7, xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='3-DOF w/o LT', legend_loc='upper left')
plotter.t_sequence = data_3DOFLT['t_sequence']
plotter.x_arch = data_3DOFLT['x_arch']
plotter.plotActualStateProgress(0, 6, '-', color=CL['RED'], xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='3-DOF LT', legend_loc='upper left')
plotter.t_sequence = data_7DOFnoLT['t_sequence']
plotter.x_arch = data_7DOFnoLT['x_arch']
plotter.plotActualStateProgress(0, 10, '-', color=CL['BLU']*0.7, xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='7-DOF w/o LT', legend_loc='upper left')
plotter.t_sequence = data_7DOFLT['t_sequence']
plotter.x_arch = data_7DOFLT['x_arch']
plotter.plotActualStateProgress(0, 10, '-', color=CL['BLU'], xlabel='Curve length ($\%$)', ylabel='$\delta$ ($rad/s$)', legend='7-DOF LT', legend_loc='lower left')
plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/four_models/Torque comparison of four models.pdf')

plt.show()
