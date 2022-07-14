import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],})
fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255,
      'WHT': np.array([255, 255, 255])/255,}


""" detect function """
def detect(l1, l2):
    sign1 = ((l2[0][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[0][0] - l1[1][0]) * (l1[0][1] - l1[1][1])) * \
        ((l2[1][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[1][0] - l1[1][0]) * (l1[0][1] - l1[1][1]))
    sign2 = ((l1[0][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[0][0] - l2[1][0]) * (l2[0][1] - l2[1][1])) * \
        ((l1[1][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[1][0] - l2[1][0]) * (l2[0][1] - l2[1][1]))
    print('sign1: ', sign1)
    print('sign2: ', sign2)
    criterion = np.arctan(sign1 / sign2)
    if criterion > - np.pi / 2 and criterion < np.pi:
        return 'not intersected'
    else:
        return 'intersected'


""" test scenario """
# 1
# l1 = np.array([[0, 0], [4, 4]])
# l2 = np.array([[0, 4], [1, 3]])

# 2
l1 = np.array([[-1, 0], [1, 0]])
l2 = np.array([[0, 2], [0, -2]])


print(detect(l1, l2))


""" Plot """

ax.plot(l1[:,0], l1[:,1], '.-', color=CL['BLU'], label='l1', linewidth=3, markersize=8, markeredgewidth=3)
ax.plot(l2[:,0], l2[:,1], '.-', color=CL['RED'], label='l2', linewidth=3, markersize=8, markeredgewidth=3)

ax.axis('equal')
# ax.set_xlim([-100, 100])
# ax.set_ylim([-100, 100])
# ax.xaxis.set_ticks([])
# ax.yaxis.set_ticks([])
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
# plt.pause(0.01)  # 暂停
ax.legend(loc='lower right', fontsize=15)  # 标签位置
ax.grid(linestyle='--')
plt.tight_layout()
# plt.savefig("figure/xx.svg")
plt.show()