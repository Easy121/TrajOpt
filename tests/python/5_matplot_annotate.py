# Annotating with Text with Box
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5), dpi=80)

t = ax.text(
    0, 0, "Direction", ha="center", va="center", rotation=0, size=15,
    bbox=dict(boxstyle="circle,pad=0.3", fc="cyan", ec="b", lw=2))

plt.show()