""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
Dual-track 7-DOF vehicle dynamics model with combined slip Magic Formula tire model
"""


import numpy as np
import casadi as ca
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


class Dual7dof:
    def __init__(self, ref, config, param) -> None:
        # from ref
        self.bl = np.asarray(ref["bl"])  # +
        self.br = np.asarray(ref["br"])  # -
        
        # from config
        self.nx = config["nx"]
        self.nu = config["nu"]
        self.mu = config["mu"]
        self.mu_c = config["mu_c"]
        
        # from param
        self.setParameters(param)
        
        ############################################################
        # Model Definition #########################################
        ############################################################
        
        """ Declare decision variables """
        # use scaling for decisions variables
        
        # scaled state vector
        vx_n    = ca.SX.sym('vx')
        vy_n    = ca.SX.sym('vy')
        dpsi_n  = ca.SX.sym('dpsi')
        omega_fl_n = ca.SX.sym('omega_fl')
        omega_fr_n = ca.SX.sym('omega_fr')
        omega_rl_n = ca.SX.sym('omega_rl')
        omega_rr_n = ca.SX.sym('omega_rr')
        n_n     = ca.SX.sym('n')
        chi_n   = ca.SX.sym('chi')
        delta_n = ca.SX.sym('delta')
        T_n     = ca.SX.sym('T')
        # the scaling for state vector (same as their maximum value)
        self.vx_s    = 20
        self.vy_s    = 1
        self.dpsi_s  = 2
        self.omega_s = 80
        self.n_s     = 3
        self.chi_s   = 1
        self.delta_s = 0.5
        self.T_s     = 200
        # state vector with normal scale
        vx    = vx_n * self.vx_s
        vy    = vy_n * self.vy_s
        dpsi  = dpsi_n * self.dpsi_s
        omega_fl = omega_fl_n * self.omega_s
        omega_fr = omega_fr_n * self.omega_s
        omega_rl = omega_rl_n * self.omega_s
        omega_rr = omega_rr_n * self.omega_s
        omega_i  = ca.vertcat(omega_fl, omega_fr, omega_rl, omega_rr)  # vector
        n     = n_n * self.n_s
        chi   = chi_n * self.chi_s
        delta = delta_n * self.delta_s
        T     = T_n * self.T_s
        delta_i = ca.vertcat(delta, delta, 0.0, 0.0)  # vector
        T_i     = ca.vertcat(0.0, 0.0, T/2.0, T/2.0)  # vector
        # formulate array
        x   = ca.vertcat(vx_n, vy_n, dpsi_n, 
                         omega_fl_n, omega_fr_n, omega_rl_n, omega_rr_n, 
                         n_n, chi_n, delta_n, T_n)
        self.x_s = np.array([self.vx_s, self.vy_s, self.dpsi_s, 
                            self.omega_s, self.omega_s, self.omega_s, self.omega_s, 
                            self.n_s, self.chi_s, self.delta_s, self.T_s])
        
        # scaled input vector
        ddelta_n = ca.SX.sym('ddelta')
        dT_n     = ca.SX.sym('dT')
        # the scaling for input vector (same as their maximum value)
        self.ddelta_s = 2
        self.dT_s     = 3000
        # input vector with normal scale
        ddelta = ddelta_n * self.ddelta_s
        dT     = dT_n * self.dT_s
        # formulate array
        u = ca.vertcat(ddelta_n, dT_n)
        self.u_s = np.array([self.ddelta_s, self.dT_s])
        
        # additional 
        kappa = ca.SX.sym('kappa')  # curvature
        ax = ca.SX.sym('ax')  # acceleration
        ay = ca.SX.sym('ay')  # acceleration
        
        ############################################################
        # Vehicle Model ############################################
        ############################################################
        
        # Wheel velocity in vehicle coordinate frame
        vx_i = vx - self.param.vehicle.y * dpsi
        vy_i = vy + self.param.vehicle.x * dpsi

        # Wheel velocity in wheel coordinate frame
        vxw_i = vx_i * ca.cos(delta_i) + vy_i * ca.sin(delta_i)
        vyw_i = - vx_i * ca.sin(delta_i) + vy_i * ca.cos(delta_i)

        kappa_i = (omega_i * self.param.wheel.R - vxw_i) / vxw_i
        alpha_i = ca.atan(vyw_i / vxw_i)
        
        # Drag force
        FD = - 0.5 * self.param.vehicle.Cd * self.param.vehicle.rho * self.param.vehicle.A * vx * vx
        # Front and rear down force.
        Fd_i = - 0.5 * self.param.vehicle.Cl * self.param.vehicle.rho * self.param.vehicle.A * vx * vx 
        Fres = FD
        
        # Load Transfer
        Fz_i = ca.SX.zeros(4)
        Fz_i[0] = ((self.param.vehicle.mt * self.param.vehicle.g * self.param.vehicle.lr \
                            - self.param.vehicle.mt * ax * self.param.vehicle.h \
                            + FD * self.param.vehicle.h - Fd_i[0] * self.param.vehicle.l) / self.param.vehicle.l) \
                            * (0.5 - (ay * self.param.vehicle.h)/(self.param.vehicle.g * self.param.vehicle.wf))

        Fz_i[1] = ((self.param.vehicle.mt * self.param.vehicle.g * self.param.vehicle.lr \
                            - self.param.vehicle.mt * ax * self.param.vehicle.h \
                            + FD * self.param.vehicle.h - Fd_i[0] * self.param.vehicle.l) / self.param.vehicle.l) \
                            * (0.5 + (ay * self.param.vehicle.h)/(self.param.vehicle.g * self.param.vehicle.wf))

        Fz_i[2] = ((self.param.vehicle.mt * self.param.vehicle.g * self.param.vehicle.lf \
                            + self.param.vehicle.mt * ax * self.param.vehicle.h \
                            - FD * self.param.vehicle.h - Fd_i[1] * self.param.vehicle.l) / self.param.vehicle.l) \
                            * (0.5 - (ay * self.param.vehicle.h)/(self.param.vehicle.g * self.param.vehicle.wr))

        Fz_i[3] = ((self.param.vehicle.mt * self.param.vehicle.g * self.param.vehicle.lf \
                            + self.param.vehicle.mt * ax * self.param.vehicle.h \
                            - FD * self.param.vehicle.h - Fd_i[1] * self.param.vehicle.l) / self.param.vehicle.l) \
                            * (0.5 + (ay * self.param.vehicle.h)/(self.param.vehicle.g * self.param.vehicle.wr))
        
        # Longitudinal
        Fxw_i = self.param.wheel.mu * ca.cos(  # Combined slip part
            self.param.wheel.Cxa * ca.atan(
            self.param.wheel.Bxa * alpha_i - self.param.wheel.Exa * ( \
            self.param.wheel.Bxa * alpha_i - ca.atan(self.param.wheel.Bxa * alpha_i)))) * \
            Fz_i * self.param.wheel.Dx * ca.sin(  # Fx0 (without combined slip)
            self.param.wheel.Cx * ca.atan(
            self.param.wheel.Bx * kappa_i - self.param.wheel.Ex * (
            self.param.wheel.Bx * kappa_i - ca.atan(self.param.wheel.Bx * kappa_i))))

        # Lateral
        Fyw_i = self.param.wheel.mu * ca.cos(  # Combined slip part
            self.param.wheel.Cyk * ca.atan(
            self.param.wheel.Byk * kappa_i - self.param.wheel.Eyk * ( \
            self.param.wheel.Byk * kappa_i - ca.atan(self.param.wheel.Byk * kappa_i)))) * \
            Fz_i * self.param.wheel.Dy * ca.sin(  # Fy0 (without combined slip)
            self.param.wheel.Cy * ca.atan(
            self.param.wheel.By * alpha_i - self.param.wheel.Ey * (
            self.param.wheel.By * alpha_i - ca.atan(self.param.wheel.By * alpha_i))))

        # In vehicle frame
        Fx_i = Fxw_i * ca.cos(delta_i) - Fyw_i * ca.sin(delta_i)
        Fy_i = Fxw_i * ca.sin(delta_i) + Fyw_i * ca.cos(delta_i)
        
        """ model descriptions """
        # convert to s-based dynamics instead of t-based
        dt    = (1 - n * kappa) / (vx * ca.cos(chi) - vy * ca.sin(chi))  # 1/ds or dt/ds integrate to get t
        
        dvx        = dt * (ca.sum1(Fx_i) + Fres) / self.param.vehicle.mt + dpsi * vy
        dvy        = dt * ca.sum1(Fy_i) / self.param.vehicle.mt - dpsi * vx
        ddpsi      = dt * ca.sum1((Fy_i * self.param.vehicle.x - Fx_i * self.param.vehicle.y)) / self.param.vehicle.Iz
        domega_i_s = dt * (T_i - Fxw_i * self.param.wheel.R) / self.param.wheel.J
        dn         = dt * (vx * ca.sin(chi) + vy * ca.cos(chi))
        dchi       = dt * dpsi - kappa
        ddelta_s   = dt * ddelta
        dT_s       = dt * dT
        
        # with respect to t
        dvx_t      = (ca.sum1(Fx_i) + Fres) / self.param.vehicle.mt + dpsi * vy
        dvy_t      = ca.sum1(Fy_i) / self.param.vehicle.mt - dpsi * vx
        ddpsi_t    = ca.sum1((Fy_i * self.param.vehicle.x - Fx_i * self.param.vehicle.y)) / self.param.vehicle.Iz
        domega_i_t = (T_i - Fxw_i * self.param.wheel.R) / self.param.wheel.J
        dn_t       = (vx * ca.sin(chi) + vy * ca.cos(chi))
        dchi_t     = dpsi - kappa / dt
        ddelta_t   = ddelta
        dT_t       = dT
        
        # for load transfer
        ax_out = dvx_t - dpsi * vy
        ay_out = dvy_t + dpsi * vx

        """ Model equations """
        dx = ca.vertcat(dvx, dvy, ddpsi, domega_i_s, dn, dchi, ddelta_s, dT_s) / self.x_s
        dx_t = ca.vertcat(dvx_t, dvy_t, ddpsi_t, domega_i_t, dn_t, dchi_t, ddelta_t, dT_t)
        # Car Info
        carinfo = ca.vertcat(Fz_i, Fxw_i, Fyw_i)

        """ Objective term """
        # L = dt
        L = 5 * dt * dt + 4 * ddelta * ddelta + 0.000001 * dT * dT
        # L = 5 * dt + 2 * ddelta * ddelta + 0.00001 * T * T

        """ Functions """
        # Continuous time dynamics (the real one is kept in self.f)
        self.f_f = ca.Function('f', [x, u, kappa, ax, ay], [dx, L, dt, ax_out, ay_out], 
                               ['x', 'u', 'kappa', 'ax', 'ay'], ['dx', 'L', 'dt', 'ax_out', 'ay_out'])
        # Time derivative information (dstate)
        self.f_d = ca.Function('f_d', [x, u, kappa], [dx_t], ['x', 'u', 'kappa'], ['dx_t'])
        # Car Info (forces)
        self.f_carinfo = ca.Function('f_carinfo', [x, u], [carinfo], ['x', 'u'], ['carinfo'])
        
        ############################################################
        # Init and Guesses #########################################
        ############################################################
        
        # init
        self.ax = 0.0
        self.ay = 0.0
        self.state_init  = [1.0 / self.vx_s, 0.0, 0.0, 
                            1.0 / self.param.wheel.R / self.omega_s, 1.0 / self.param.wheel.R / self.omega_s,
                            1.0 / self.param.wheel.R / self.omega_s, 1.0 / self.param.wheel.R / self.omega_s, 
                            (self.bl[0]+self.br[0])/2 / self.n_s, 0.0, 0.0, 0.0
        ]
        # state
        self.state_guess = [5.0 / self.vx_s, 0.0, 0.0, 
                            25 / self.omega_s, 25 / self.omega_s, 25 / self.omega_s, 25 / self.omega_s, 
                            (self.bl[0]+self.br[0])/2 / self.n_s, 0.0, 0.0, 0.0
        ]
        # input
        self.input_guess = [0.0] * self.nu
    
    ############################################################
    # Parameters ###############################################
    ############################################################
    
    def f(self, x, u, kappa):
        dx, L, dt, ax_temp, ay_temp = self.f_f(x, u, kappa, self.ax, self.ay)
        # self.ax = ax_temp
        # self.ay = ay_temp
        return dx, L, dt
    
    ############################################################
    # Parameters ###############################################
    ############################################################
    
    def setParameters(self, param):
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
        
        self.param = Param(
            vehicle=vehicle,
            steering=steering,
            wheel=wheel,
            driving=driving,
        )
        
    ############################################################
    # Bounds ###################################################
    ############################################################
    # * dimension check
    
    def getStateMin(self, k):
        state_min = [
            0.0 / self.vx_s,  # vx
            -np.inf,  # vy 
            -np.inf,  # dpsi
            -np.inf,  # omega
            -np.inf,  # omega
            -np.inf,  # omega
            -np.inf,  # omega
            self.br[k] / self.n_s,  # n
            -np.inf,  # chi
            -0.43 / self.delta_s,  # delta
            -200 / self.T_s,  # T
        ]
        return state_min
    
    def getStateMax(self, k):
        state_max = [
            10.0 / self.vx_s,  # vx
            np.inf,  # vy 
            np.inf,  # dpsi
            np.inf,  # omega
            np.inf,  # omega
            np.inf,  # omega
            np.inf,  # omega
            self.bl[k] / self.n_s,  # n
            np.inf,  # chi
            0.43 / self.delta_s,  # delta
            200 / self.T_s,  # T
        ]
        return state_max
    
    def getInputMin(self, k):
        input_min = [
            -1.414 / self.ddelta_s,  # ddelta
            -3000 / self.dT_s,  # dT
        ]
        return input_min
    
    def getInputMax(self, k):
        input_max = [
            1.414 / self.ddelta_s,  # ddelta
            3000 / self.dT_s,  # dT
        ]
        return input_max
