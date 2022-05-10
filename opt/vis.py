import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.transforms as mtransforms
from matplotlib.patches import Ellipse
import yaml

# Structure of the full plot
# e.g., I want to plot 5 plots, the it will generate a full plot with 2-row-3-column subplots
PlotNumConvert = {
    # two plots are reserved for time box plot
    1: [1, 1],
    2: [1, 2],
    3: [2, 2],
    4: [2, 2],
    5: [2, 3],
    6: [2, 3],
    7: [3, 3],
    8: [3, 3],
    9: [3, 3],
    10: [3, 4],
    11: [3, 4],
    12: [3, 4],
    13: [4, 4],
    14: [4, 4],
    15: [4, 4],
    16: [4, 4],
}

CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,}


class Plotter():
    def __init__(self,problem, simulator, num_of_plots, width_ratios=[1, 9], figsize=(18, 8), mode='normal', interval=1):
        self.getGeneralAxes(num_of_plots, width_ratios, figsize)
        
        # set mode to string other than 'normal', 
        # so that you can test stand-alone methods without having data
        self.mode = mode
        
        if self.mode == 'normal':
            self.simulator  = simulator
            self.MPC_step   = simulator.MPC_step
            self.ctr_step   = simulator.ctr_step
            self.sim_step   = simulator.sim_step
            self.t_total    = simulator.t_total
            self.t_sequence = simulator.t_sequence
            self.t_ctrl_sequence = simulator.t_ctrl_sequence
            # for large dataset, plotting every one of them is time-consuming
            # interval should be an integar, the data amount will be divided by this number
            # e.g. interval=2 omit half of data. 
            # this will affect plotActualXY, plotActualdState, plotActualState, plotActualInput, plotReference
            self.interval = interval

            # control problem
            self.problem    = problem
            self.H     = self.problem.H
            self.Hc    = self.problem.Hc
            self.n     = self.problem.n
            self.m     = self.problem.m

            # simulation problem
            self.problem_sim   = simulator.problem_sim
            # self.H_sim = self.problem_sim.H  # not needed
            self.n_sim = self.problem_sim.n
            self.m_sim = self.problem_sim.m
            
            # Actual Archives
            self.dx_arch = simulator.dx_arch.reshape(-1, self.n_sim)
            if simulator.carinfo_arch.size != 0:
                self.carinfo_arch = simulator.carinfo_arch.reshape(np.shape(self.dx_arch)[0], -1)
            else: self.carinfo_arch = np.empty([0, 0])
            self.x_arch  = simulator.x_arch.reshape(-1, self.n_sim)
            self.u_arch  = simulator.u_arch.reshape(-1, self.m_sim)
            # Predictive Archives
            self.xPred_arch = simulator.xPred_arch.reshape(-1, self.n * (self.H + 1))
            self.uOpt_arch  = simulator.uOpt_arch.reshape(-1, self.m * self.Hc)
            # Solver addition
            if simulator.is_solver_addition == True:
                self.solver_addition_arch = simulator.solver_addition_arch.reshape(np.shape(self.uOpt_arch)[0], -1)
            else: self.solver_addition_arch = np.empty([0, 0])
            # flag
            self.is_ref_initialized = False
        
        else:
            self.simulator  = None
            self.MPC_step   = None
            self.ctr_step   = None
            self.sim_step   = None
            self.t_total    = None
            self.t_sequence = None
            self.t_ctrl_sequence = None
            
            self.interval = interval

            self.problem    = None
            self.H     = None
            self.Hc    = None
            self.n     = None
            self.m     = None

            self.problem_sim   = None
            self.n_sim = None
            self.m_sim = None
            
            self.dx_arch = np.empty([0, 0])
            self.carinfo_arch = np.empty([0, 0])
            self.x_arch  = np.empty([0, 0])
            self.u_arch  = np.empty([0, 0])
            self.xPred_arch = np.empty([0, 0])
            self.uOpt_arch  = np.empty([0, 0])
            self.solver_addition_arch = np.empty([0, 0])
            
            self.is_ref_initialized = False
    
    def getGeneralAxes(self, num_of_plots, width_ratios, figsize):
        """ Return the axes for plotting """
        if width_ratios[0] == 0:
            # construct plot
            combination = PlotNumConvert.get(num_of_plots, 'Not a valid key')
            fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=80)    
            fig.canvas.manager.set_window_title('GMPCPlatform')
            self.ax = []  # only one dimensional
            index = 1
            for row in range(combination[0]):
                for col in range(combination[1]):
                    ax_temporary = fig.add_subplot(combination[0], combination[1], index)
                    # mandatory setting
                    ax_temporary.grid(linestyle='--')
                    self.ax.append(ax_temporary) 
                    index += 1
        else:
            fig = plt.figure(constrained_layout=True, figsize=figsize, dpi=80)
            fig.canvas.manager.set_window_title('GMPCPlatform')
            gs0 = fig.add_gridspec(1, 2, width_ratios=width_ratios)
            # two grids always reserved for time 
            gs00 = gs0[0].subgridspec(2, 1)
            self.ax_time1 = fig.add_subplot(gs00[0, 0])
            self.ax_time2 = fig.add_subplot(gs00[1, 0])

            combination = PlotNumConvert.get(num_of_plots, 'Not a valid key')
            if type(combination) == str:
                print(combination)
                return 

            # user-defined grids always reserved for data
            gs01 = gs0[1].subgridspec(combination[0], combination[1])

            self.ax = []  # only one dimensional
            for row in range(combination[0]):
                for col in range(combination[1]):
                    ax_temporary = fig.add_subplot(gs01[row, col])
                    # mandatory setting
                    ax_temporary.grid(linestyle='--')
                    self.ax.append(ax_temporary)

    #################################
    # Box Plot ######################
    #################################

    def setTime(self, solver_time_array, simulation_time_array):
        self.plotBoxDistribution(self.ax_time1, solver_time_array, 'Solver')
        self.plotBoxDistribution(self.ax_time2, simulation_time_array, 'Simulation')

    def plotBoxDistribution(self, ax, time_array, xlabel):
        time_fontsize = 15
        ax.boxplot(time_array)
        ax.set_xlabel(xlabel, fontsize=time_fontsize)
        ax.set_ylabel('Time distribution (ms)', fontsize=time_fontsize)
        ax.set_ylim(bottom=0)
        ax.grid(linestyle='--')

    ###################################
    # Utility #########################
    ###################################
    
    def getPlotRange(self, data):
        """get square XY range"""
        data = np.array(data)
        # the first column is x, the second column is 
        x_max = np.max(data[:, 0])
        x_min = np.min(data[:, 0])
        x_range = x_max - x_min
        x_mid = (x_max + x_min) / 2

        y_max = np.max(data[:, 1])
        y_min = np.min(data[:, 1])
        y_range = y_max - y_min
        y_mid = (y_max + y_min) / 2

        if x_range > y_range:
            plot_range = x_range
            y_max = y_mid + plot_range / 2
            y_min = y_mid - plot_range / 2
        else:
            plot_range = y_range
            x_max = x_mid + plot_range / 2
            x_min = x_mid - plot_range / 2

        return [[x_min, x_max], [y_min, y_max]]
    
    def plotRaceCar(self, ax, Z, transform):
        w = 1.22
        img_w = 400
        img_h = 198
        
        extent = [-img_w/img_h*w/2, img_w/img_h*w/2, -w/2, w/2]
        im = ax.imshow(Z, interpolation='none',
                    origin='lower',
                    extent=extent, clip_on=True)

        trans_data = transform + ax.transData
        im.set_transform(trans_data)

        # # display intended extent of the image
        # x1, x2, y1, y2 = im.get_extent()
        # ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
        #         transform=trans_data)
        
    def plotFrictionEllipse(self, ax_index, Fm  # list, [Fxm, Fym]
                            , mu  # road adhesion coefficient
                            , color='k', alpha=0.08, legend='Friction ellipse'):
        # Friction ellipse
        ellipse = Ellipse(xy=(0, 0), width=Fm[0]*mu*2, height=Fm[1]*mu*2, ec='None', fc=color, alpha=alpha, label=legend)
        self.ax[ax_index].add_patch(ellipse)

    ###################################
    # Full Plot #######################
    ###################################

    ###################################
    # Building Blocks Plot ############
    ###################################

    # Track
    def plotTrack(self, ax_index, arg, w=1.22, color=False):
        if arg == 'DLC':  # double lane change
            start_1 = 0
            end_1 = 12
            start_2 = end_1
            end_2 = start_2 + 13.5
            start_3 = end_2
            end_3 = start_3 + 11
            start_4 = end_3
            end_4 = start_4 + 12.5
            start_5 = end_4
            end_5 = start_5 + 12

            bl_1 = (1.1*w+0.25)/2
            br_1 = -(1.1*w+0.25)/2
            bl_3 = (1.1*w+0.25)/2+1+(w+1)
            br_3 = (1.1*w+0.25)/2+1
            bl_5 = np.maximum((1.3*w+0.25), 3)/2
            br_5 = -np.maximum((1.3*w+0.25), 3)/2
            # plot the lane change boundary
            # 1
            self.ax[ax_index].plot([start_1, end_1], [bl_1]*2, '-', color=CL['RED']*0.6, label='Boundary', linewidth=3)
            self.ax[ax_index].plot([start_1, end_1], [br_1]*2, '-', color=CL['RED']*0.6, linewidth=3)
            # 3
            self.ax[ax_index].plot([start_3, end_3], [bl_3]*2, '-', color=CL['RED']*0.6, linewidth=3)
            self.ax[ax_index].plot([start_3, end_3], [br_3]*2, '-', color=CL['RED']*0.6, linewidth=3)
            # 5
            self.ax[ax_index].plot([start_5, end_5], [bl_5]*2, '-', color=CL['RED']*0.6, linewidth=3)
            self.ax[ax_index].plot([start_5, end_5], [br_5]*2, '-', color=CL['RED']*0.6, linewidth=3)
        else:
            with open(arg) as stream:
                data = yaml.safe_load(stream)
                lx = data['lx']
                ly = data['ly']
                rx = data['rx']
                ry = data['ry']
                if color is False:
                    self.ax[ax_index].plot(lx, ly, '.', color=CL['RED'],
                        label='Left cones', linewidth=3, markersize=8)
                    self.ax[ax_index].plot(rx, ry, '.', color=CL['BLU'],
                        label='Right cones', linewidth=3, markersize=8)
                else:
                    self.ax[ax_index].plot(lx, ly, '.', color=color,
                        label='Cones', linewidth=3, markersize=8)
                    self.ax[ax_index].plot(rx, ry, '.', color=color, linewidth=3, markersize=8)
                self.ax[ax_index].axis('equal')
            
    def plotReference(self, ax_index, path, format='--', color=CL['ORA'], legend_fontsize=10, legend_loc='upper right', interval=10):
        # ! time-consuming
        if self.is_ref_initialized == False:
            with open(path) as stream:
                self.ref = yaml.safe_load(stream)
                self.is_ref_initialized = True
        self.ax[ax_index].plot(self.ref['x'][0::interval], self.ref['y'][0::interval], format, color=color,
            label='Reference', linewidth=2, markersize=8)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
            
    def plotReferenceVx(self, ax_index, s_index, mpcsolver, path=None, format='--', color=CL['ORA'], legend_fontsize=10, legend_loc='upper right'):
        if self.is_ref_initialized == False:
            with open(path) as stream:
                self.ref = yaml.safe_load(stream)
                self.is_ref_initialized = True
        s_data = self.solver_addition_arch[:, s_index]
        vx = []
        for index in range(s_data.size):
            track_index = mpcsolver.solver.calcIndex(s_data[index])
            vx.append(self.ref['vx'][track_index])
        vx = np.asarray(vx)
        self.ax[ax_index].plot(self.t_ctrl_sequence, vx, format, color=color, 
            label='Reference', linewidth=2, markersize=8)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        
    def plotReferenceVxProgress(self, ax_index, s_index, mpcsolver, path=None, format='--', color=CL['ORA'], legend_fontsize=10, legend_loc='upper right'):
        if self.is_ref_initialized == False:
            with open(path) as stream:
                self.ref = yaml.safe_load(stream)
                self.is_ref_initialized = True
        s_data = self.solver_addition_arch[:, s_index]
        vx = []
        for index in range(s_data.size):
            track_index = mpcsolver.solver.calcIndex(s_data[index])
            vx.append(self.ref['vx'][track_index])
        vx = np.asarray(vx)
        self.ax[ax_index].plot(self.t_ctrl_sequence / self.t_ctrl_sequence[-1] * 100, vx, format, color=color, 
            label='Reference', linewidth=2, markersize=8)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
            
    # Solver addition
    def plotSolverAddition(self, 
                           ax_index, 
                           index, 
                           format,  # format of points and lines
                           color=CL['RED'],
                           xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='',
                           label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                           linewidth=3, markersize=8, markeredgewidth=3):
        data = self.solver_addition_arch[:, index]
        self.ax[ax_index].plot(self.t_ctrl_sequence, data, format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_ctrl_sequence[-1]])

    # XY
    def plotActualXY(self, 
                     ax_index, 
                     X_index, 
                     Y_index, 
                     Psi_index,
                     format,  # format of points and lines
                     color=CL['BLK'],
                     xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Actual',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3, set_plot_range=False, axis_equal=True, race_car=True):
        data_X = self.x_arch[:, X_index]
        data_Y = self.x_arch[:, Y_index]
        data_Psi = self.x_arch[:, Psi_index]
        
        # race car plot
        if race_car is True:
            img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'race_car.png')
            img = mpimg.imread(img_path)
            self.plotRaceCar(self.ax[ax_index], img, mtransforms.Affine2D().rotate_deg(np.rad2deg(data_Psi[0])).translate(data_X[0], data_Y[0]))
            self.plotRaceCar(self.ax[ax_index], img, mtransforms.Affine2D().rotate_deg(np.rad2deg(data_Psi[-1])).translate(data_X[-1], data_Y[-1]))
            
        # orignal plot
        self.ax[ax_index].plot(data_X[0::self.interval], data_Y[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

        data = np.hstack((data_X.reshape(-1, 1),data_Y.reshape(-1, 1)))
        if set_plot_range == True:
            self.plot_range = self.getPlotRange(data)
            self.ax[ax_index].set_xlim(self.plot_range[0])
            self.ax[ax_index].set_ylim(self.plot_range[1])
        else:
            if axis_equal == True:
                self.ax[ax_index].axis('equal')
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        
    def plotActualXYFrame(self, 
                     ax_index, 
                     X_index, 
                     Y_index, 
                     Psi_index,
                     num_of_frames,
                     format,  # format of points and lines
                     color=CL['BLK'],
                     xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Actual',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3, set_plot_range=False, axis_equal=True):
        # plot the vehicle locations in frames, like a movie
        data_X = self.x_arch[:, X_index]
        data_Y = self.x_arch[:, Y_index]
        data_Psi = self.x_arch[:, Psi_index]
        
        # race car plot
        img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'race_car.png')
        img = mpimg.imread(img_path)
        
        frames = np.linspace(0, data_X.size, num_of_frames)
        for frame in frames:
            frame_index = int(frame)
            if frame_index == data_X.size:
                frame_index -= 1
            self.plotRaceCar(self.ax[ax_index], 
                             img, 
                             mtransforms.Affine2D().rotate_deg(
                                 np.rad2deg(data_Psi[frame_index])).translate(
                                     data_X[frame_index], data_Y[frame_index]))
        
        # orignal plot
        self.ax[ax_index].plot(data_X[0::self.interval], data_Y[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)

        data = np.hstack((data_X.reshape(-1, 1),data_Y.reshape(-1, 1)))
        if set_plot_range == True:
            self.plot_range = self.getPlotRange(data)
            self.ax[ax_index].set_xlim(self.plot_range[0])
            self.ax[ax_index].set_ylim(self.plot_range[1])
        else:
            if axis_equal == True:
                self.ax[ax_index].axis('equal')
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
    
    def plotPredictedXY(self,
                        ax_index,
                        X_index,
                        Y_index,
                        format,  # format of points and lines
                        color=CL['BLU'],
                        xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Predictive',
                        label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                        linewidth=3, markersize=8, markeredgewidth=3, first_predict=0, predict_interval=10):
        # include the last one
        for ctr_progress in list(range(first_predict, np.shape(self.xPred_arch)[0], predict_interval)) + [np.shape(self.xPred_arch)[0]-1]: 
            data = self.xPred_arch[ctr_progress, :]
            data = data.reshape(-1, self.n)
            # for the state specified by state_index
            data_X = data[:, X_index].flatten()
            data_Y = data[:, Y_index].flatten()
            # only add legend to the first curve
            if ctr_progress == first_predict:
                self.ax[ax_index].plot(data_X, data_Y, format, color=color, label=legend, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)
            else:
                self.ax[ax_index].plot(data_X, data_Y, format, color=color, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)

        # self.ax[ax_index].set_xlim(self.plot_range[0])
        # self.ax[ax_index].set_ylim(self.plot_range[1])
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        
    def plotPredictedsn2XY(self,
                           ax_index,
                           s_index,
                           n_index,
                           mpcsolver,
                           format,  # format of points and lines
                           color=CL['BLU'],
                           xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Predictive',
                           label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                           linewidth=3, markersize=8, markeredgewidth=3, first_predict=0, predict_interval=10):
        # use the fastindex in mpcsolver to find x,y coordinates from s,n 
        # include the last one
        for ctr_progress in list(range(first_predict, np.shape(self.xPred_arch)[0], predict_interval)) + [np.shape(self.xPred_arch)[0]-1]: 
            data = self.xPred_arch[ctr_progress, :]
            data = data.reshape(-1, self.n)
            # for the state specified by state_index
            data_s = data[:, s_index].flatten()
            data_n = data[:, n_index].flatten()
            data_X = []
            data_Y = []
            for i in range(0, data_s.size):
                pose = mpcsolver.solver.calcCartesPose2(np.array([data_s[i], data_n[i]]))
                data_X.append(pose[0])
                data_Y.append(pose[1])
            data_X = np.asarray(data_X)
            data_Y = np.asarray(data_Y)
            # only add legend to the first curve
            if ctr_progress == 0:
                self.ax[ax_index].plot(data_X, data_Y, format, color=color, label=legend, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)
            else:
                self.ax[ax_index].plot(data_X, data_Y, format, color=color, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)

        # self.ax[ax_index].set_xlim(self.plot_range[0])
        # self.ax[ax_index].set_ylim(self.plot_range[1])
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        
    def plotOptXY(self, 
                  ax_index, 
                  n_index, 
                  ref,  # reference data
                  format,  # format of points and lines
                  color=CL['BLK'],
                  xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Optimized',
                  label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                  linewidth=3, markersize=8, markeredgewidth=3, set_plot_range=False):
        x = np.append(np.asarray(ref["x"]), ref["x"][0])
        y = np.append(np.asarray(ref["y"]), ref["y"][0])
        theta = np.append(np.asarray(ref["theta"]), ref["theta"][0])
        data_n = self.x_arch[:, n_index]
        data_X = x - data_n * np.sin(theta)
        data_Y = y + data_n * np.cos(theta)

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
        
    def plotOptXYFrame(self, 
                       ax_index, 
                       n_index, 
                       chi_index, 
                       num_of_frames,  # number of car frames
                       ref,  # reference data
                       format,  # format of points and lines
                       color=CL['BLK'],
                       xlabel='$X$ ($m$)', ylabel='$Y$ ($m$)', legend='Optimized',
                       label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                       linewidth=3, markersize=8, markeredgewidth=3, set_plot_range=False):
        x = np.append(np.asarray(ref["x"]), ref["x"][0])
        y = np.append(np.asarray(ref["y"]), ref["y"][0])
        theta = np.append(np.asarray(ref["theta"]), ref["theta"][0])
        data_n = self.x_arch[:, n_index]
        data_X = x - data_n * np.sin(theta)
        data_Y = y + data_n * np.cos(theta)
        data_chi = self.x_arch[:, chi_index]
        data_Psi = theta + data_chi
        
        # race car plot
        img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'source/race_car.png')
        img = mpimg.imread(img_path)

        frames = np.linspace(0, data_X.size, num_of_frames)
        for frame in frames:
            frame_index = int(frame)
            if frame_index == data_X.size:
                frame_index -= 1
            self.plotRaceCar(self.ax[ax_index], 
                             img, 
                             mtransforms.Affine2D().rotate_deg(
                                 np.rad2deg(data_Psi[frame_index])).translate(
                                     data_X[frame_index], data_Y[frame_index]))
        
        # orignal plot
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

    # dstate
    def plotActualdState(self, 
                         ax_index, 
                         state_index, 
                         format,  # format of points and lines
                         color=CL['RED'],
                         xlabel='', ylabel='', legend='Actual',
                         label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                         linewidth=3, markersize=8, markeredgewidth=3, omit_start=100):
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        data = self.dx_arch[:, state_index]
        self.ax[ax_index].plot(self.t_sequence[omit_start::self.interval], data[omit_start::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
        
    def plotActualAx(self, 
                     ax_index,
                     dvx_index,  # dstate
                     vy_index,  # state
                     dpsi_index,  # state
                     format,  # format of points and lines
                     color=CL['RED'],
                     xlabel='', ylabel='', legend='Actual',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3, omit_start=100):
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        dvx_data  = self.dx_arch[omit_start::self.interval, dvx_index]
        vy_data   = self.x_arch[omit_start::self.interval, vy_index]
        dpsi_data = self.x_arch[omit_start::self.interval, dpsi_index]
        ax = dvx_data - dpsi_data * vy_data
        self.ax[ax_index].plot(self.t_sequence[omit_start::self.interval], ax, format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
    
    def plotActualAy(self, 
                     ax_index,
                     dvy_index,  # dstate
                     vx_index,  # state
                     dpsi_index,  # state
                     format,  # format of points and lines
                     color=CL['RED'],
                     xlabel='', ylabel='', legend='Actual',
                     label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                     linewidth=3, markersize=8, markeredgewidth=3, omit_start=100):
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        dvy_data  = self.dx_arch[omit_start::self.interval, dvy_index]
        vx_data   = self.x_arch[omit_start::self.interval, vx_index]
        dpsi_data = self.x_arch[omit_start::self.interval, dpsi_index]
        ay = dvy_data + dpsi_data * vx_data
        self.ax[ax_index].plot(self.t_sequence[omit_start::self.interval], ay, format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])

    # CarInfo
    def plotActualCarInfo(self, 
                          ax_index, 
                          info_index, 
                          format,  # format of points and lines
                          color=CL['RED'],
                          xlabel='', ylabel='', legend='Actual',
                          label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                          linewidth=3, markersize=8, markeredgewidth=3, omit_start=100):
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        data = self.carinfo_arch[:, info_index]
        self.ax[ax_index].plot(self.t_sequence[omit_start::self.interval], data[omit_start::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
        
    def plotActualTireForce(self, 
                            ax_index, 
                            Fx_index, 
                            Fy_index, 
                            format,  # format of points and lines
                            color=CL['RED'],
                            xlabel='$F_x$ ($N$)', ylabel='$F_y$ ($N$)', legend='Actual',
                            label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                            linewidth=3, markersize=3, markeredgewidth=3, omit_start=100, interval=10):
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        Fx_data = self.carinfo_arch[:, Fx_index]
        Fy_data = self.carinfo_arch[:, Fy_index]
        self.ax[ax_index].plot(Fx_data[omit_start::interval], Fy_data[omit_start::interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].axis('equal')
        
    def plotActualTireForceDless(self, 
                                 ax_index, 
                                 Fx_index, 
                                 Fy_index, 
                                 Fz_index, 
                                 format,  # format of points and lines
                                 color=CL['RED'],
                                 xlabel='$F_x / F_z$', ylabel='$F_y / F_z$', legend='Actual',
                                 label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                                 linewidth=3, markersize=3, markeredgewidth=3, omit_start=100, interval=10):
        """ Dimensionless tire force plot """
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        Fx_data = self.carinfo_arch[omit_start::interval, Fx_index]
        Fy_data = self.carinfo_arch[omit_start::interval, Fy_index]
        Fz_data = self.carinfo_arch[omit_start::interval, Fz_index]
        Fx_Dless_data = Fx_data / Fz_data
        Fy_Dless_data = Fy_data / Fz_data
        self.ax[ax_index].plot(Fx_Dless_data, Fy_Dless_data, format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].axis('equal')
        
    def plotActualTireForceDlessEnd(self, 
                                    ax_index, 
                                    Fx_index, 
                                    Fy_index, 
                                    Fz_index, 
                                    format,  # format of points and lines
                                    color=CL['RED'],
                                    xlabel='$F_x / F_z$', ylabel='$F_y / F_z$',
                                    label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                                    linewidth=3, markersize=12, markeredgewidth=3, zorder=5):
        """ Plot the end index of dimensionless tire force """
        # omit_start: the starting data are vibrating and soaring high, therefore omitted
        Fx_data = self.carinfo_arch[-1, Fx_index]
        Fy_data = self.carinfo_arch[-1, Fy_index]
        Fz_data = self.carinfo_arch[-1, Fz_index]
        Fx_Dless_data = Fx_data / Fz_data
        Fy_Dless_data = Fy_data / Fz_data
        self.ax[ax_index].plot(Fx_Dless_data, Fy_Dless_data, format, zorder=zorder, color=color,
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].axis('equal')

    # State
    def plotActualState(self, 
                        ax_index, 
                        state_index, 
                        format,  # format of points and lines
                        color=CL['RED'],
                        xlabel='', ylabel='', legend='Actual',
                        label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                        linewidth=3, markersize=8, markeredgewidth=3):
        data = self.x_arch[:, state_index]
        self.ax[ax_index].plot(self.t_sequence[0::self.interval], data[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
        
    def plotActualStateProgress(self, 
                                ax_index, 
                                state_index, 
                                format,  # format of points and lines
                                color=CL['RED'],
                                xlabel='Curve length ($\%$)', ylabel='', legend='Actual',
                                label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                                linewidth=3, markersize=8, markeredgewidth=3):
        data = self.x_arch[:, state_index]
        self.ax[ax_index].plot(self.t_sequence[0::self.interval] / self.t_sequence[0::self.interval][-1] * 100, data[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, 100])

    def plotPredictedState(self,
                           ax_index,
                           state_index,
                           format,  # format of points and lines
                           color=CL['BLU'],
                           xlabel='', ylabel='', legend='Predictive',
                           label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                           linewidth=3, markersize=8, markeredgewidth=3, first_predict=0, predict_interval=10):
        # t_horizon = np.linspace(self.MPC_step, self.H * self.MPC_step, self.H)
        t_horizon = np.linspace(0, self.H * self.MPC_step, self.H + 1)

        # include the last one
        for ctr_progress in list(range(first_predict, np.shape(self.xPred_arch)[0], predict_interval)) + [np.shape(self.xPred_arch)[0]-1]: 
            data = self.xPred_arch[ctr_progress, :]
            data = data.reshape(-1, self.n)
            # for the state specified by state_index
            data_state = data[:, state_index].flatten()
            t_state    = ctr_progress * self.ctr_step + t_horizon
            # only add legend to the first curve
            if ctr_progress == first_predict:
                self.ax[ax_index].plot(t_state, data_state, format, color=color, label=legend, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)
            else:
                self.ax[ax_index].plot(t_state, data_state, format, color=color, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)

        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])

    # Input
    def plotActualInput(self, 
                        ax_index, 
                        input_index, 
                        format,  # format of points and lines
                        color=CL['ORA'],
                        xlabel='', ylabel='', legend='Actual',
                        label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                        linewidth=3, markersize=8, markeredgewidth=3):
        data = self.u_arch[:, input_index]
        self.ax[ax_index].plot(self.t_sequence[0::self.interval], data[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
        
    def plotActualInputProgress(self, 
                        ax_index, 
                        input_index, 
                        format,  # format of points and lines
                        color=CL['ORA'],
                        xlabel='Curve length ($\%$)', ylabel='', legend='Actual',
                        label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                        linewidth=3, markersize=8, markeredgewidth=3):
        data = self.u_arch[:, input_index]
        self.ax[ax_index].plot(self.t_sequence[0::self.interval] / self.t_sequence[0::self.interval][-1] * 100, data[0::self.interval], format, color=color, label=legend, 
            linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth)
        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, 100])

    def plotPredictedInput(self,
                           ax_index,
                           input_index,
                           format,  # format of points and lines
                           color=CL['BLU'],
                           xlabel='', ylabel='', legend='Predictive',
                           label_fontsize=15, legend_fontsize=10, legend_loc='upper right',
                           linewidth=3, markersize=8, markeredgewidth=3, first_predict=0, predict_interval=10):
        # predictive input generates before simulation, it is in line with simulation time
        t_horizon = np.linspace(0, (self.Hc - 1) * self.MPC_step, self.Hc)
        
        # include the last one
        for ctr_progress in list(range(first_predict, np.shape(self.xPred_arch)[0], predict_interval)) + [np.shape(self.xPred_arch)[0]-1]: 
            data = self.uOpt_arch[ctr_progress, :]
            data = data.reshape(-1, self.m)
            # for the state specified by state_index
            data_input = data[:, input_index].flatten()
            t_state    = ctr_progress * self.ctr_step + t_horizon
            # only add legend to the first curve
            if ctr_progress == first_predict:
                self.ax[ax_index].plot(t_state, data_input, format, color=color, label=legend, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)
            else:
                self.ax[ax_index].plot(t_state, data_input, format, color=color, 
                    linewidth=linewidth, markersize=markersize, markeredgewidth=markeredgewidth, alpha=0.3)

        self.ax[ax_index].set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax[ax_index].set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax[ax_index].legend(loc=legend_loc, fontsize=legend_fontsize)
        self.ax[ax_index].set_xlim([0, self.t_sequence[-1]])
        
    ###################################
    # Save ############################
    ###################################

    def save(self, path):
        if self.mode == 'normal':
            if self.carinfo_arch.size != 0 and self.solver_addition_arch.size != 0:
                # The complete form
                np.savez(path, 
                        H = self.H,
                        Hc = self.Hc,
                        n = self.n,
                        m = self.m,
                        MPC_step = self.MPC_step,
                        ctr_step = self.ctr_step,
                        sim_step = self.sim_step,
                        t_sequence = self.t_sequence,
                        t_ctrl_sequence = self.t_ctrl_sequence,
                        x_arch=self.x_arch, 
                        dx_arch=self.dx_arch, 
                        u_arch=self.u_arch,
                        xPred_arch=self.xPred_arch, 
                        uOpt_arch=self.uOpt_arch,
                        carinfo_arch=self.carinfo_arch,
                        solver_addition_arch=self.solver_addition_arch,
                )
            elif self.carinfo_arch.size == 0 and self.solver_addition_arch.size != 0:
                # no carinfo
                np.savez(path, 
                        H = self.H,
                        Hc = self.Hc,
                        n = self.n,
                        m = self.m,
                        MPC_step = self.MPC_step,
                        ctr_step = self.ctr_step,
                        sim_step = self.sim_step,
                        t_sequence = self.t_sequence,
                        t_ctrl_sequence = self.t_ctrl_sequence,
                        x_arch=self.x_arch, 
                        dx_arch=self.dx_arch, 
                        u_arch=self.u_arch,
                        xPred_arch=self.xPred_arch, 
                        uOpt_arch=self.uOpt_arch,
                        solver_addition_arch=self.solver_addition_arch,
                )
            elif self.carinfo_arch.size != 0 and self.solver_addition_arch.size == 0:
                # no solver addition
                np.savez(path, 
                        H = self.H,
                        Hc = self.Hc,
                        n = self.n,
                        m = self.m,
                        MPC_step = self.MPC_step,
                        ctr_step = self.ctr_step,
                        sim_step = self.sim_step,
                        t_sequence = self.t_sequence,
                        t_ctrl_sequence = self.t_ctrl_sequence,
                        x_arch=self.x_arch, 
                        dx_arch=self.dx_arch, 
                        u_arch=self.u_arch,
                        xPred_arch=self.xPred_arch, 
                        uOpt_arch=self.uOpt_arch,
                        carinfo_arch=self.carinfo_arch,
                )
            else:
                # basic form
                np.savez(path, 
                        H = self.H,
                        Hc = self.Hc,
                        n = self.n,
                        m = self.m,
                        MPC_step = self.MPC_step,
                        ctr_step = self.ctr_step,
                        sim_step = self.sim_step,
                        t_sequence = self.t_sequence,
                        t_ctrl_sequence = self.t_ctrl_sequence,
                        x_arch=self.x_arch, 
                        dx_arch=self.dx_arch, 
                        u_arch=self.u_arch,
                        xPred_arch=self.xPred_arch, 
                        uOpt_arch=self.uOpt_arch,
                )
        else:
            # for use in TrajOpt
            np.savez(path, 
                    t_sequence = self.t_sequence,
                    x_arch=self.x_arch, 
                    dx_arch=self.dx_arch, 
                    u_arch=self.u_arch,
                    carinfo_arch=self.carinfo_arch,
            )
        
