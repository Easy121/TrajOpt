import os 
import yaml
import numpy as np
from dataclasses import dataclass


@dataclass
class Vehicle:
    mt: float
    Iz: float
    g:  float

    l:       float
    lf:      float
    lr:      float
    h:       float
    wf:      float
    wf_half: float
    wr:      float
    wr_half: float

    x: np.ndarray
    y: np.ndarray

    Cd: float
    Cl: np.ndarray
    rho: float
    A:   float

@dataclass
class Steering:
    sr: float
    st: float
    sa: float
    sk: float
    rp: float
    D:  float

    DP_init: np.ndarray
    l1_init: np.ndarray
    l2_init: np.ndarray
    beta_init: np.ndarray

    SR: float
    VR: float
    p: float
    omega_max: float
    omega_ope: float

@dataclass
class Wheel:
    J: float
    R: float

    mu: float

    Bx: float
    Cx: float
    Dx: float
    Ex: float

    By: float
    Cy: float
    Dy: float
    Ey: float

    Bxa: float
    Cxa: float
    Exa: float
    Byk: float
    Cyk: float
    Eyk: float
    
@dataclass
class Driving:
    a_max: float
    T_max: float
    T_con: float
    VR: float
    
@dataclass
class Param:
    vehicle: Vehicle 
    steering: Steering
    wheel: Wheel
    driving: Driving


path_to_param = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'param.yaml')
with open(path_to_param) as stream:
    param = yaml.safe_load(stream)

vehicle = Vehicle(
    mt=param['vehicle']['mt'],
    Iz=param['vehicle']['Iz'],
    g=param['vehicle']['g'],
    l=param['vehicle']['l'],
    lf=param['vehicle']['lf'],
    lr=param['vehicle']['lr'],
    h=param['vehicle']['h'],
    wf=param['vehicle']['wf'],
    wf_half=param['vehicle']['wf_half'],
    wr=param['vehicle']['wr'],
    wr_half=param['vehicle']['wr_half'],
    x=np.asarray(param['vehicle']['x']),
    y=np.asarray(param['vehicle']['y']),
    Cd=param['vehicle']['Cd'],
    Cl=np.asarray(param['vehicle']['Cl']),
    rho=param['vehicle']['rho'],
    A=param['vehicle']['A'],
)

steering = Steering(
    sr=param['steering']['sr'],
    st=param['steering']['st'],
    sa=param['steering']['sa'],
    sk=param['steering']['sk'],
    rp=param['steering']['rp'],
    D=param['steering']['D'],
    DP_init=np.asarray(param['steering']['DP_init']),
    l1_init=np.asarray(param['steering']['l1_init']),
    l2_init=np.asarray(param['steering']['l2_init']),
    beta_init=np.asarray(param['steering']['beta_init']),
    SR=param['steering']['SR'],
    VR=param['steering']['VR'],
    p=param['steering']['p'],
    omega_max=param['steering']['omega_max'],
    omega_ope=param['steering']['omega_ope'],
)

wheel = Wheel(
    J=param['wheel']['J'],
    R=param['wheel']['R'],
    mu=param['wheel']['mu'],
    Bx=param['wheel']['Bx'],
    Cx=param['wheel']['Cx'],
    Dx=param['wheel']['Dx'],
    Ex=param['wheel']['Ex'],
    By=param['wheel']['By'],
    Cy=param['wheel']['Cy'],
    Dy=param['wheel']['Dy'],
    Ey=param['wheel']['Ey'],
    Bxa=param['wheel']['Bxa'],
    Cxa=param['wheel']['Cxa'],
    Exa=param['wheel']['Exa'],
    Byk=param['wheel']['Byk'],
    Cyk=param['wheel']['Cyk'],
    Eyk=param['wheel']['Eyk'],
)

driving = Driving(
    a_max=param['driving']['a_max'],
    T_max=param['driving']['T_max'],
    T_con=param['driving']['T_con'],
    VR=param['driving']['VR'],
)

param_all = Param(
    vehicle=vehicle,
    steering=steering,
    wheel=wheel,
    driving=driving,
)

print(vehicle)
print('')
print(steering)
print('')
print(wheel)
print('')
print(driving)
print('')
print(param_all)