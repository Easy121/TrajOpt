import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"],})
CL = {'BLU': np.array([0, 114, 189])/255,
      'LBLU': np.array([173, 216, 230])/255,
      'RED': np.array([217, 83, 25])/255,
      'LRED': np.array([255, 204, 203])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


""" Init """
step = 1.0  # m
# vehicle 
v_max = 2.5  # m/s
delta_max = 0.75  # rad
# time
t = step / v_max
t_int = 0.01


""" Sim """
# model parameters
lw = 2.8  # wheelbase
lf = 0.96  # front overhang length
lr = 0.929  # rear overhang length
w = 1.924  # width


def f_continuous(x, u):
    """ model descriptions """
    # X     = x[0]
    # Y     = x[1]
    Psi   = x[2]
    v     = x[3]
    delta = x[4]
    
    # a = u[0]
    # ddelta = u[1]
    
    dX   = v * np.cos(Psi)
    dY   = v * np.sin(Psi)
    dPsi = v * np.tan(delta) / lw
    # a
    # ddelta
    
    return np.append(np.array([dX, dY, dPsi]), u)


def rungeKutta4(f, x, u, t):
    k1 = f(x, u)
    k2 = f(x + t * k1 / 2.0, u)
    k3 = f(x + t * k2 / 2.0, u)
    k4 = f(x + t * k3, u)
    dx = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return x + t * dx


# only one simulation is needed
x_init = np.array([0.0, 0.0, 0.0, v_max, delta_max])
x = x_init
u = np.array([0.0, 0.0])

for i in range(int(t/t_int)):
    x = rungeKutta4(f_continuous, x, u, t_int)
    
# information from the simulation
increm_Psi = x[2]
increm_ang = np.arctan(x[1] / x[0])
# increm_len = np.sqrt(x[0]**2 + x[1]**2)
increm_len = np.sqrt(x[0]**2 + x[1]**2)


""" Dict """
Psi2increm = dict()
vis_left   = []
vis_center = []
vis_right  = []

for i in range(360):
    Psi = np.deg2rad(i)
    increm_pos = [
        [increm_len * np.cos(Psi + increm_ang), increm_len * np.sin(Psi + increm_ang), int(np.rad2deg(increm_Psi))],
        [step * np.cos(Psi), step * np.sin(Psi), 0],
        [increm_len * np.cos(Psi - increm_ang), increm_len * np.sin(Psi - increm_ang), int(- np.rad2deg(increm_Psi))],
    ]
    Psi2increm[i] = increm_pos
    
    # vis
    vis_left.append(increm_pos[0])
    vis_center.append(increm_pos[1])
    vis_right.append(increm_pos[2])

# path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'map_Psi2increm.npy')
# np.save(path_to_save, Psi2increm)

vis_left   = np.asarray(vis_left)
vis_center = np.asarray(vis_center)
vis_right  = np.asarray(vis_right)

# test
# path_to_save = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'map_Psi2increm.npy')
# Psi2increm = np.load(path_to_save, allow_pickle='TRUE').item()
# print(Psi2increm[90])


""" Vis """
# fig and coordinate
fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=80)

locations = [[0, 0], [step, 0], [-step, 0], [0, step], [0, -step]]
for location in locations:
    grid = patches.Rectangle((0,0), step, step, fill=False, linewidth=3, edgecolor=CL['LBLU'])
    transform = mtransforms.Affine2D().translate(-step/2, -step/2).translate(location[0], location[1]) + ax.transData
    grid.set_transform(transform)
    ax.add_patch(grid)

ax.scatter(vis_center[:, 0], vis_center[:, 1])

plt.tight_layout()
ax.axis('equal')
plt.show()