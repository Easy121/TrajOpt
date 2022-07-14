import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
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


fig = plt.figure()
ax = fig.add_subplot(111)

# model parameters
lw = 2.8  # wheelbase
lf = 0.96  # front overhang length
lr = 0.929  # rear overhang length
w = 1.924  # width

X = 4
Y = 5
Psi = 0.34
delta = 0.34

# vehicle body patch
rectangle = patches.Rectangle((0,0), lw+lf+lr, w, fill=False, linewidth=3, edgecolor=CL['BLU'], alpha=0.50, label='Vehicle body')
transform = mtransforms.Affine2D().translate(-lr, -w/2).rotate_around(0, 0, Psi).translate(X, Y) + ax.transData
rectangle.set_transform(transform)
ax.add_patch(rectangle)
# vehicle orientation arrow and origin
arrow_length = 1
ax.arrow(X, Y, arrow_length*np.cos(Psi), arrow_length*np.sin(Psi), width=0.1, color=CL['RED'])
circle = patches.Circle((X, Y), 0.1, color=CL['RED'], label='Start and end pose')
ax.add_patch(circle)

plt.xlim(-20, 60)
plt.ylim(-20, 60)

plt.grid(True)
ax.axis('equal')
plt.legend(loc='upper right')
plt.show()