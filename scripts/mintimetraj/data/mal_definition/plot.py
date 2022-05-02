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
path_to_ref_right = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/reference_right_sparse.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../../referenceline/data/track.yaml')
path_to_data_mal = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'mal_defined.npz')
path_to_data_cor = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'correct.npz')

data_mal = np.load(path_to_data_mal)
data_cor = np.load(path_to_data_cor)
# Import reference
with open(path_to_ref_center, 'r') as stream:
    ref_center = yaml.safe_load(stream)
with open(path_to_ref_right, 'r') as stream:
    ref_right = yaml.safe_load(stream)

""" Vis """
# plotter = Plotter(None, None, 12, width_ratios=[0, 0], figsize=(24, 12), mode='debug', interval=1)
# plotter.t_sequence = data['t_sequence']
# plotter.x_arch = data['x_arch']
# plotter.u_arch = data['u_arch']
# plotter.dx_arch = data['dx_arch']
# plotter.carinfo_arch = data['carinfo_arch']

# 1. path
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(10, 5), mode='debug', interval=1)
# plotter.t_sequence = data_mal['t_sequence']
# plotter.x_arch = data_mal['x_arch']
# plotter.plotTrack(0, path_to_track)
# plotter.plotReference(0, path_to_ref_center, color=CL['BLK'], interval=1)
# plotter.plotOptXY(0, 3, ref_center, '-', color=CL['ORA'], legend='Mal-defined model', legend_loc='upper right')
# plotter.t_sequence = data_cor['t_sequence']
# plotter.x_arch = data_cor['x_arch']
# plotter.plotOptXY(0, 3, ref_center, '-', color=CL['BLU'], legend='Correct model', legend_loc='upper right')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/mal_defined_model/Path results of mal-defined and correct models.pdf')

# # 2. yaw rate
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(10, 5), mode='debug', interval=1)
# plotter.t_sequence = data_mal['t_sequence']
# plotter.x_arch = data_mal['x_arch']
# plotter.plotActualState(0, 2, '-', color=CL['ORA'], xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='Mal-defined model', legend_loc='upper left')
# plotter.t_sequence = data_cor['t_sequence']
# plotter.x_arch = data_cor['x_arch']
# plotter.plotActualState(0, 2, '-', color=CL['BLU'], xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', legend='Correct model', legend_loc='upper left')
# plt.savefig('/Users/gzj/FILES/SYNCHRON/T/Direction_Doctor/Motion Planning and Control of 2-IWD Autonomous Electric Racing Car under Limit Conditions/My Work/BachelorThesis/figure/results/time_optimal_solutions/mal_defined_model/Yaw rate results of mal-defined and correct models.pdf')

# # 3. n compare
# plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(10, 5), mode='debug', interval=1)
# plotter.t_sequence = data_right['t_sequence']
# plotter.x_arch = data_right['x_arch']
# plotter.plotActualState(0, 3, '-', color=CL['ORA'], xlabel='Time ($s$)', ylabel='$n$ ($m$)', legend='Right Reference', legend_loc='upper left')
# plotter.t_sequence = data_center['t_sequence']
# plotter.x_arch = data_center['x_arch']
# plotter.plotActualState(0, 3, '-', color=CL['BLU'], xlabel='Time ($s$)', ylabel='$n$ ($m$)', legend='Center Reference', legend_loc='upper left')
# plt.savefig(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'n comparison for different references.pdf'))

plt.show()
