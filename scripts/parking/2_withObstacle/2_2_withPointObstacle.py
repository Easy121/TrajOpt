""" 
Test of automatic parking
* kinematic model
* with point obstacle
"""



from opt.calc import *
from opt.opt import *
from opt.opt_parking import *
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


""" Construct Problem """

""" init and ref """
# # 1 straight path TROUBLE
# init = [0.0, 0.0, 0.0]
# ref = [30.0, 0.0, 0.0]
# obs = [[15.0, 0]]

# 2 diagonal path SUCCESS
init = [0.0, 0.0, np.pi/4]
ref = [8.0, 8.0, np.pi/4]
obs = [[4, 4]]

# # 2-1 extreme diagonal path TROUBLE
# init = [0.0, 0.0, np.pi/4]
# ref = [20.0, 1.0, np.pi/4]
# obs = [[10, 0.5]]

# # 3 diamond
# init = [0.0, 0.0, 0.0]
# ref = [30.0, 0.0, 0.0]
# obs = [np.array([[14, 0], [15, 1]]), np.array([[15, 1], [16, 0]]), np.array([[16, 0], [15, -1]]), np.array([[15, -1], [14, 0]])]
# # obs = [np.array([[0, 5], [10, -2]])]

# object
# traj = Parking_Opt(init, ref, None)
traj = Parking_Opt(init, ref, obs)


""" Optimize """

traj.optimize()


""" Plot """
# * vis.py is the replicate of visual.py from GMPCPlatform, this is an efficient workaround, please keep updated

vis_t0 = time.perf_counter()

# parse data
plotter = Plotter(None, None, 6, width_ratios=[0, 0], figsize=(20, 8), mode='debug', interval=1)
plotter.t_sequence = traj.t_opt
plotter.x_arch = traj.x_opt
plotter.u_arch = traj.u_opt
plotter.dx_arch = traj.dx_opt

""" state plot """
# plotter.plotTrack(0, path_to_track)
# plotter.plotReference(0, path_to_reference, color=CL['BLU'], interval=1)

# plotter.plotParkingXY(0, 0, 1, 2, legend_loc='upper right')
plotter.plotParkingXYFrame(0, 0, 1, 2, traj.N, legend_loc='upper right')
plotter.plotActualState(1, 3, '-', xlabel='Time ($s$)', ylabel='$v$ ($m/s$)', legend_loc='lower right')
plotter.plotActualState(2, 4, '-', xlabel='Time ($s$)', ylabel='$\delta$ ($rad$)', legend_loc='lower right')

plotter.plotActualdState(3, 2, '-', omit_start=0, xlabel='Time ($s$)', ylabel='$\dot{\psi}$ ($rad/s$)', color=CL['RED'], legend_loc='upper right')

""" input plot """
plotter.plotActualInput(4, 0, '-', xlabel='Time ($s$)', ylabel='$a$ ($m/s^2$)', color=CL['ORA'], legend_loc='upper right')
plotter.plotActualInput(5, 1, '-', xlabel='Time ($s$)', ylabel='$\dot{\delta}$ ($rad/s$)', color=CL['ORA'], legend_loc='upper left')

vis_t = time.perf_counter() - vis_t0
print("[TIME] Visualization takes: %.3f s" % vis_t) # CPU seconds elapsed (floating point)
plt.show()

# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/7dof_2.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/7dof_2_closed.npz'))
# plotter.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/7dof_2_closed_low_mu.npz'))


