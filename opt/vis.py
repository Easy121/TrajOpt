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
    def __init__(self, ref, track, traj, num_of_plots, figsize=(18, 8), mode='normal', interval=1):
        self.x = np.append(np.asarray(ref["x"]), ref["x"][0])
        self.y = np.append(np.asarray(ref["y"]), ref["y"][0])
        self.s = np.append(np.asarray(ref["s"]), ref["s"][0])
        self.kappa = np.append(np.asarray(ref["kappa"]), ref["kappa"][0])
        self.theta = np.append(np.asarray(ref["theta"]), ref["theta"][0])
        self.lx = np.asarray(track['lx'])
        self.ly = np.asarray(track['ly'])
        self.rx = np.asarray(track['rx'])
        self.ry = np.asarray(track['ry'])
        
        # get traj info
        self.traj = traj
        self.x_opt = traj.x_opt
        self.u_opt = traj.u_opt
        self.t_opt = traj.t_opt
        self.t_total = self.t_opt[-1]
        
        # construct plot
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
        
    # Track
    def plotReference(self, ax_index):
        self.ax[ax_index].plot(self.x, self.y, '-', color=CL['BLU'],
            label='Reference', linewidth=1, markersize=8)
        self.ax[ax_index].axis('equal')

    # XY
    def plotOptXY(self, 
                  ax_index, 
                  n_index, 
                  format,  # format of points and lines
                  color=CL['BLK'],
                  xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Optimized',
                  label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                  linewidth=3, markersize=8, markeredgewidth=3, set_plot_range=False):
        data_n = self.x_opt[n_index, :]
        data_X = self.x - data_n * np.sin(self.theta)
        data_Y = self.y + data_n * np.cos(self.theta)

        self.ax[ax_index].plot(data_X[0::self.interval], data_Y[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

        data = np.hstack((data_X.reshape(-1, 1),data_Y.reshape(-1, 1)))
        if set_plot_range is True:
            self.plot_range = self.getPlotRange(data)
            self.ax[ax_index].set_xlim(self.plot_range[0])
            self.ax[ax_index].set_ylim(self.plot_range[1])
        else:
            self.ax[ax_index].axis('equal')
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)

    # State
    def plotOptState(self, 
                     ax_index, 
                     state_index, 
                     format,  # format of points and lines
                     color=CL['BLU'],
                     xlabel='', ylabel='', legend='Optimized',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3):
        data = self.x_opt[state_index, :]
        self.ax[ax_index].plot(self.t_opt[0::self.interval], data[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_total])
        
    def plotOptInput(self, 
                     ax_index, 
                     input_index, 
                     format,  # format of points and lines
                     color=CL['ORA'],
                     xlabel='', ylabel='', legend='Optimized',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3):
        data = self.u_opt[input_index, :]
        self.ax[ax_index].plot(self.t_opt[0::self.interval], np.append(0.0, data[0::self.interval]), format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_total])