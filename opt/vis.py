""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


import numpy as np
import matplotlib.pyplot as plt 
import yaml


# Structure of the full plot
# e.g., I want to plot 5 plots, the it will generate a full plot with 2-row-3-column subplots
PlotNumConvert = {
    # two plots are reserved for time box plot
    1: [1, 2],
    2: [1, 2],
    3: [2, 2],
    4: [2, 2],
    5: [2, 3],
    6: [2, 3],
    7: [3, 3],
    8: [3, 3],
    9: [3, 3],
    10: [4, 3],
    11: [4, 3],
    12: [4, 3],
}

CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,}


class Plotter():
    def __init__(self, ref, track, num_of_plots, figsize=(18, 8), mode='normal', interval=1):
        self.x = np.asarray(ref["x"])
        self.y = np.asarray(ref["y"])
        self.s = np.asarray(ref["s"])
        self.kappa = np.asarray(ref["kappa"])
        self.lx = np.asarray(track['lx'])
        self.ly = np.asarray(track['ly'])
        self.rx = np.asarray(track['rx'])
        self.ry = np.asarray(track['ry'])
        
        combination = PlotNumConvert.get(num_of_plots, 'Not a valid key')
        fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=80)    
        self.ax = []  # only one dimensional
        index = 1
        for row in range(combination[0]):
            for col in range(combination[1]):
                ax_temporary = fig.add_subplot(combination[0], combination[1], index)
                # mandatory setting
                ax_temporary.grid(linestyle='--')
                self.ax.append(ax_temporary) 
                index += 1
        self.interval = interval
        
    ###################################
    # Building Blocks Plot ############
    ###################################

    # Track
    def plotTrack(self, ax_index):
        self.ax[ax_index].plot(self.lx, self.ly, '.', color=CL['RED'],
            label='Left cones', linewidth=2, markersize=8)
        self.ax[ax_index].plot(self.rx, self.ry, '.', color=CL['BLU'],
            label='Right cones', linewidth=2, markersize=8)
        self.ax[ax_index].axis('equal')
