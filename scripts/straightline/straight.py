""" 
The code for generating staight reference
"""


import yaml
import os
import numpy as np


# 100 m
total_length = 200


""" Reference """
# normally 0.05m as interval for online control 
num = 4000
output = {
    'x': np.linspace(0, total_length, num=num).tolist(),
    'y': [0.0]*num,
    's': np.linspace(0, total_length, num=num).tolist(),
    'kappa': [0.0]*num,
    'theta': [0.0]*num,
    'bl': [4.5]*num,
    'br': [-1.5]*num,
    'vx': [10.0]*num,
}
path_to_output = os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'data/' + 'reference_straight' + '.yaml')
with open(path_to_output, 'w') as stream:
    yaml.dump(output, stream)
    
# """ Track """
# num = 10

# # a normal track should be 3.75 m in width
# # for the current single-seat vehicle, the width is set to 3 m
# output = {
#     'lx': np.linspace(0, total_length, num=num).tolist(),  # left x
#     'ly': [4.5]*num,
#     'rx': np.linspace(0, total_length, num=num).tolist(),  # right x
#     'ry': [-1.5]*num,
# }
# path_to_output = os.path.join(os.path.abspath(
#     os.path.dirname(__file__)), 'data/' + 'track' + '.yaml')
# with open(path_to_output, 'w') as stream:
#     yaml.dump(output, stream)
