""" 
Generate reference path from trajectory optimization for vehicle tracking
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


""" Data """

data = np.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../7dof_2.npz'))
# data = np.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../7dof_noa.npz'))


""" Process """

