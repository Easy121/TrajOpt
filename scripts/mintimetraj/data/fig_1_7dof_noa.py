""" 
Figures of 7dof_noa
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
path_to_reference = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/reference_center_sparse.yaml')
path_to_track = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../referenceline/data/track.yaml')
path_to_data = os.path.join(os.path.abspath(os.path.dirname(__file__)), '7dof_noa.npz')

data = np.load(path_to_data)
# Import reference
with open(path_to_reference, 'r') as stream:
    ref = yaml.safe_load(stream)

""" Vis """
# plotter = Plotter(None, None, 12, width_ratios=[0, 0], figsize=(24, 12), mode='debug', interval=1)
# plotter.t_sequence = data['t_sequence']
# plotter.x_arch = data['x_arch']
# plotter.u_arch = data['u_arch']
# plotter.dx_arch = data['dx_arch']
# plotter.carinfo_arch = data['carinfo_arch']

# 1. Vehicle trajectory
plotter = Plotter(None, None, 1, width_ratios=[0, 0], figsize=(15, 10), mode='debug', interval=1)
plotter.t_sequence = data['t_sequence']
plotter.x_arch = data['x_arch']
plotter.plotTrack(0, path_to_track)
plotter.plotReference(0, path_to_reference, color=CL['BLU'], interval=1)
plotter.plotOptXYFrame(0, 7, 8, 60, ref, '-', legend_loc='upper right')
plotter.ax[0].set(xlim=[-10,30], ylim=[-45, -5])

plt.show()
