""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
Vehicle model based on bi_3dof_ddelta.py
Features load transfer, nonlinear tire model and friction ellipse constraints
"""


import numpy as np
import casadi as ca


class B3dcLT:
    def __init__(self, ref, config) -> None:
        # from ref
        self.bl = np.asarray(ref["bl"])  # +
        self.br = np.asarray(ref["br"])  # -
        
        # from config
        self.nx = config["nx"]
        self.nu = config["nu"]
        self.mu = config["mu"]
        self.mu_c = config["mu_c"]
        
        ############################################################
        # Model Definition #########################################
        ############################################################
        
        """ Declare decision variables """
        # use scaling for decisions variables
        
        # scaled state vector
        vx_n    = ca.SX.sym('vx')
        vy_n    = ca.SX.sym('vy')
        dpsi_n  = ca.SX.sym('dpsi')
        n_n     = ca.SX.sym('n')
        chi_n   = ca.SX.sym('chi')
        delta_n = ca.SX.sym('delta')
        T_n     = ca.SX.sym('T')
        # the scaling for state vector (same as their maximum value)
        self.vx_s    = 20
        self.vy_s    = 1
        self.dpsi_s  = 2
        self.n_s     = 3
        self.chi_s   = 1
        self.delta_s = 0.5
        self.T_s     = 200
        # state vector with normal scale
        vx    = vx_n * self.vx_s
        vy    = vy_n * self.vy_s
        dpsi  = dpsi_n * self.dpsi_s
        n     = n_n * self.n_s
        chi   = chi_n * self.chi_s
        delta = delta_n * self.delta_s
        T     = T_n * self.T_s
        # formulate array
        x   = ca.vertcat(vx_n, vy_n, dpsi_n, n_n, chi_n, delta_n, T_n)
        self.x_s = np.array([self.vx_s, self.vy_s, self.dpsi_s, self.n_s, self.chi_s, self.delta_s, self.T_s])
        
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
        
        """ model parameters """
        By = 9.882
        Cy = 1.111
        Dy = -2.559
        Ey = 0.2949
        Dx = 2.339
        
        R  = 0.2186
        m  = 230
        g  = 9.81
        Iz = 138.53
        h  = 0.34;  # CoG vertical position
        lf = 0.858
        lr = 0.702
        l  = lr + lf
        
        # Load Transfer
        ax = T / R * 0.94 / m;  # simplified and ignore lateral force
        Fzf = m * (g * lr - ax * h) / l
        Fzr = m * (g * lf + ax * h) / l
        
        # intermediate value
        # the torque also drives wheels
        Fx  = T / R * 0.94  # 94% due to acceleration of wheels
        # no need to multiply two because Fzf already twice
        alpha_f = ca.atan((vy + dpsi * lf) / vx) - delta
        Fyf = self.mu * Fzf * Dy * ca.sin(Cy * ca.atan(By * alpha_f - Ey * (By * alpha_f - ca.atan(By * alpha_f))))
        alpha_r = ca.atan((vy - dpsi * lr) / vx)
        Fyr = self.mu * Fzr * Dy * ca.sin(Cy * ca.atan(By * alpha_r - Ey * (By * alpha_r - ca.atan(By * alpha_r))))
        
        """ model descriptions """
        # convert to s-based dynamics instead of t-based
        dt    = (1 - n * kappa) / (vx * ca.cos(chi) - vy * ca.sin(chi))  # 1/ds or dt/ds integrate to get t
        
        # with respect to s
        dvx      = dt * ((Fx - Fyf * ca.sin(delta)) / m + dpsi * vy)
        dvy      = dt * ((Fyf * ca.cos(delta) + Fyr) / m - dpsi * vx)
        ddpsi    = dt * ((Fyf * lf * ca.cos(delta) + Fx * lf * ca.sin(delta) - Fyr * lr) / Iz)
        dn       = dt * (vx * ca.sin(chi) + vy * ca.cos(chi))
        dchi     = dt * dpsi - kappa
        ddelta_s = dt * ddelta
        dT_s     = dt * dT
        
        # with respect to t
        dvx_t    = ((Fx - Fyf * ca.sin(delta)) / m + dpsi * vy)
        dvy_t    = ((Fyf * ca.cos(delta) + Fyr) / m - dpsi * vx)
        ddpsi_t  = ((Fyf * lf * ca.cos(delta) + Fx * lf * ca.sin(delta) - Fyr * lr) / Iz)
        dn_t     = (vx * ca.sin(chi) + vy * ca.cos(chi))
        dchi_t   = dpsi - kappa / dt
        ddelta_t = ddelta
        dT_t     = dT

        """ Model equations """
        dx = ca.vertcat(dvx, dvy, ddpsi, dn, dchi, ddelta_s, dT_s) / self.x_s
        dx_t = ca.vertcat(dvx_t, dvy_t, ddpsi_t, dn_t, dchi_t, ddelta_t, dT_t)
        # Car Info
        Fxf = 0.0
        Fxr = Fx
        carinfo = ca.vertcat(Fzf, Fzr, Fxf, Fxr, Fyf, Fyr)

        """ Objective term """
        # L = dt
        L = 5 * dt * dt + 4 * ddelta * ddelta + 0.000001 * dT * dT
        # L = 5 * dt + 2 * ddelta * ddelta + 0.00001 * T * T
        
        """ Constraint term """
        # recalculate ax and vertical load
        ax = (Fx - Fyf * ca.sin(delta)) / m  # ax is not dvx
        Fzf = m * (g * lr - ax * h) / l
        Fzr = m * (g * lf + ax * h) / l
        g_1 = (Fx / (self.mu_c * Dx * Fzr)) * (Fx / (self.mu_c * Dx * Fzr)) + \
            (Fyr / (self.mu_c * Dy * Fzr)) * (Fyr / (self.mu_c * Dy * Fzr))
        g = ca.vertcat(g_1)

        """ Functions """
        # Continuous time dynamics
        self.f = ca.Function('f', [x, u, kappa], [dx, L, dt], ['x', 'u', 'kappa'], ['dx', 'L', 'dt'])
        # Constraint
        self.f_g = ca.Function('f_g', [x, u], [g], ['x', 'u'], ['g'])
        # Time derivative information (dstate)
        self.f_d = ca.Function('f_d', [x, u, kappa], [dx_t], ['x', 'u', 'kappa'], ['dx_t'])
        # Car Info (forces)
        self.f_carinfo = ca.Function('f_carinfo', [x, u], [carinfo], ['x', 'u'], ['carinfo'])
        
        ############################################################
        # Inits ####################################################
        ############################################################
        # * dimension check
        
        self.state_init  = [1.0 / self.vx_s, 0.0, 0.0, (self.bl[0]+self.br[0])/2 / self.n_s, 0.0, 0.0, 0.0]

    ############################################################
    # Guesses ##################################################
    ############################################################
    # * dimension check
    
    def getStateGuess(self, k):
        return [5.0 / self.vx_s, 0.0, 0.0, (self.bl[0]+self.br[0])/2 / self.n_s, 0.0, 0.0, 0.0]
    
    def getInputGuess(self, k):
        return [0.0] * self.nu

    ############################################################
    # Bounds ###################################################
    ############################################################
    # * dimension check
    
    def getStateMin(self, k):
        state_min = [
            0.0 / self.vx_s,  # vx
            -np.inf,  # vy 
            -np.inf,  # dpsi
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
    
    def getConstraintMin(self):
        return [0.0]
    
    def getConstraintMax(self):
        return [1.0]