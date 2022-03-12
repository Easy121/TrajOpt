""" 
SIMPLE TEST
"""

import os
import sys
path_current = os.path.abspath(os.path.dirname(__file__))
path_bag = os.path.dirname(os.path.dirname(path_current))
sys.path.insert(0, path_bag) 
from opt.opt import *
from opt.calc import *

import csv
import yaml
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "DeJavu Serif",
    "font.serif": ["Computer Modern Roman"], })
fig, ax = plt.subplots(1, 4, figsize=(20, 5), dpi=80)
CL = {'BLU': np.array([0, 114, 189])/255,
      'RED': np.array([217, 83, 25])/255,
      'ORA': np.array([235, 177, 32])/255,
      'PUR': np.array([126, 172, 48])/255,
      'GRE': np.array([119, 172, 48])/255,
      'BRO': np.array([162, 20, 47])/255,
      'BLK': np.array([0, 0, 0])/255, }


""" ZEROS TEST """
# linkage = np.zeros([10, 5])
# linkage[0, 0] = 1
# print(linkage)


""" NUMPY NONZERO TEST """
# x = np.array([[0, 0, 3], [0, 5, 0], [0, 3, 0]])
# print(x)
# print(type(x.nonzero()))
# print(x.nonzero())
# print(type(x.nonzero()[0]))
# print(x.nonzero()[0])
# print(x[x.nonzero()])

# print(x[:,1].nonzero()[0])
# print(x[:,1][x[:,1].nonzero()])

# linkage = np.array([[0, 1, 3], [0, 5, 0], [0, 3, 0]])
# linkage_row = linkage[0,:]
# linkage_row_nonzero_index = linkage_row.nonzero()
# linkage_row_nonzero = linkage_row[linkage_row_nonzero_index]
# print(linkage_row)
# print(linkage_row_nonzero_index)
# print(linkage_row_nonzero)
# print(linkage_row_nonzero.shape[0])


""" MATH TEST """
x = 2
print(int(x / 2.0))
