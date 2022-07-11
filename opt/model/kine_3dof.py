""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
Kinematic bicycle model with 3 DOFs
* this model corresponds to the one stipulated in TPCAP (https://www.tpcap.net/#/rules)
"""


from re import A
import numpy as np
import casadi as ca


class Kine3dof:
    def __init__(self, init, ref, obs) -> None:
        # init and ref are lists with [X, Y, Psi]
        self.init = init
        self.ref = ref
        
        # obstacle description
        self.obs = obs
        
        # X, Y, Psi, v, delta
        self.nx = 5
        # a, ddelta
        self.nu = 2
        
        ############################################################
        # Model Definition #########################################
        ############################################################
        
        # Declare decision variables
        # use scaling for decisions variables
        
        # scaled state vector
        X_n    = ca.SX.sym('X')
        Y_n    = ca.SX.sym('Y')
        Psi_n  = ca.SX.sym('Psi')
        v_n     = ca.SX.sym('v')
        delta_n = ca.SX.sym('delta')
        # the scaling for state vector (same as their maximum value)
        self.X_s    = np.max([np.abs(self.init[0]), np.abs(self.ref[0])])
        self.Y_s    = np.max([np.abs(self.init[1]), np.abs(self.ref[1])])
        self.Psi_s  = np.pi
        self.v_s     = 2.5
        self.delta_s = 0.75
        # state vector with normal scale
        X    = X_n * self.X_s
        Y    = Y_n * self.Y_s
        Psi  = Psi_n * self.Psi_s
        v     = v_n * self.v_s
        delta = delta_n * self.delta_s
        # formulate array
        x   = ca.vertcat(X_n, Y_n, Psi_n, v_n, delta_n)
        self.x_s = np.array([self.X_s, self.Y_s, self.Psi_s, self.v_s, self.delta_s])
        
        # scaled input vector
        a_n = ca.SX.sym('a')
        ddelta_n      = ca.SX.sym('ddelta')
        # the scaling for input vector (same as their maximum value)
        self.a_s      = 1
        self.ddelta_s = 0.5
        # input vector with normal scale
        a      = a_n * self.a_s
        ddelta = ddelta_n * self.ddelta_s
        # formulate array
        u = ca.vertcat(a_n, ddelta_n)
        self.u_s = np.array([self.a_s, self.ddelta_s])
        
        # model parameters
        lw = 2.8  # wheelbase
        lf = 0.96  # front overhang length
        lr = 0.929  # rear overhang length
        w = 1.924  # width
        
        # model descriptions
        dX   = v * ca.cos(Psi)
        dY   = v * ca.sin(Psi)
        dPsi = v * ca.tan(delta) / lw
        # a
        # ddelta
        
        # Model equations
        dx = ca.vertcat(dX, dY, dPsi, a, ddelta) / self.x_s

        # Objective term
        L = 10 * (a * a + v * v * dPsi * dPsi)

        # Continuous time dynamics
        self.f = ca.Function('f', [x, u], [dx, L], ['x', 'u'], ['dx', 'L'])
        # Time derivative information (dstate)
        self.f_d = ca.Function('f_d', [x, u], [dx], ['x', 'u'], ['dx'])
        
    ############################################################
    # Guesses ##################################################
    ############################################################
    # * dimension check
    
    def getStateGuess(self, k):
        return self.init + [0.0, 0.0]
    
    def getInputGuess(self, k):
        return [0.0] * self.nu
        
    ############################################################
    # Bounds ###################################################
    ############################################################
    # * dimension check
    
    def getStateMin(self, k):
        state_min = [
            -np.inf,  # X
            -np.inf,  # Y
            -np.inf,  # Psi
            -2.5 / self.v_s,  # v
            -0.75 / self.delta_s,  # delta
        ]
        return state_min
    
    def getStateMax(self, k):
        state_max = [
            np.inf,  # X
            np.inf,  # Y 
            np.inf,  # Psi
            2.5 / self.v_s,  # v
            0.75 / self.delta_s,  # delta
        ]
        return state_max
    
    def getInputMin(self, k):
        input_min = [
            -1.0 / self.a_s,  # a
            -0.5 / self.ddelta_s,  # ddelta
        ]
        return input_min
    
    def getInputMax(self, k):
        input_max = [
            1.0 / self.a_s,  # a
            0.5 / self.ddelta_s,  # ddelta
        ]
        return input_max