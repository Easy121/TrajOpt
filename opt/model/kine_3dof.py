""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
Kinematic bicycle model with 3 DOFs
* This model corresponds to the one stipulated in TPCAP (https://www.tpcap.net/#/rules)
"""


import numpy as np
import casadi as ca


class Kine3dof:
    def __init__(self, init, ref, obs) -> None:
        # init and ref are lists with [X, Y, Psi]
        self.init_position = np.array([init[0], init[1]])
        self.init = [0.0, 0.0, init[2]]
        self.ref = [ref[0]-init[0], ref[1]-init[1], ref[2]]
        
        # obstacle description
        self.obs = obs
        if self.obs is not None:
            self.n_obs = len(obs)
        
        # X, Y, Psi, v, delta
        self.nx = 5
        # a, ddelta
        self.nu = 2
        
        """ number of intermediate state """
        # first calculate the distance, and determine N based on dis
        dis = np.floor(np.sqrt(np.square(self.ref[0]) + np.square(self.ref[1])))
        self.N = np.max([int(dis * 2), 40])
        
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
        # * 1. no scaling 
        # self.X_s    = 1
        # self.Y_s    = 1
        # * 2. different scaling (tested to be the fastest)
        self.X_s = np.max([np.abs(self.ref[0]) * 1.2, 1])
        self.Y_s = np.max([np.abs(self.ref[1]) * 1.2, 1])
        # * 3. uniform scaling 
        # self.X_s = np.max([np.abs(self.ref[0]), np.abs(self.ref[1])]) * 1.2  # 1.2 for space expansion
        # self.Y_s = self.X_s
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
        self.fc = (3 * lw + 3 * lf - lr) / 4  # front disc center from rear axle center
        self.rc = (lw + lf - 3 * lr) / 4  # rear disc center from rear axle center
        self.Rd2 = np.square((lw + lf + lr) / 4) + np.square(w / 2)   # square of disc radius
        
        # model descriptions
        dX   = v * ca.cos(Psi)
        dY   = v * ca.sin(Psi)
        dPsi = v * ca.tan(delta) / lw
        # a
        # ddelta
        
        # Model equations
        dx = ca.vertcat(dX, dY, dPsi, a, ddelta) / self.x_s

        # Objective term
        # * C3
        # * C4
        # * orient as soon as possible
        # L = 10 * (a * a + v * v * dPsi * dPsi) + \
        #     10 * delta * delta
        L = 10 * (a**2 + v**2 * dPsi**2) + \
            10 * delta**2 + \
            10 * (Psi - self.ref[2])**2

        # Continuous time dynamics
        self.f = ca.Function('f', [x, u], [dx, L], ['x', 'u'], ['dx', 'L'])
        # Time derivative information (dstate)
        self.f_d = ca.Function('f_d', [x, u], [dx], ['x', 'u'], ['dx'])
        
    ############################################################
    # Guesses ##################################################
    ############################################################
    # * dimension check
    
    def getStateGuess(self, percent):
        # return [self.ref[0] * percent, self.ref[1] * percent, self.init[2] + (self.ref[2] - self.init[2]) * percent, 0.0, 0.0]
        return self.init + [0.0, 0.0]
    
    def getInputGuess(self, k):
        return [0.0] * self.nu
    
    ############################################################
    # Obstacles ################################################
    ############################################################
    
    def f_g(self, Xk):
        g = []
        X = Xk[0] * self.X_s
        Y = Xk[1] * self.Y_s
        Psi = Xk[2] * self.Psi_s
        
        for P in self.obs:
            # front
            Xf = X + self.fc * ca.cos(Psi)
            Yf = Y + self.fc * ca.sin(Psi)
            g.append((Xf - P[0])**2 + (Yf - P[1])**2)
            
            # rear
            Xr = X + self.rc * ca.cos(Psi)
            Yr = Y + self.rc * ca.sin(Psi)
            g.append((Xr - P[0])**2 + (Yr - P[1])**2)
        
            # g.append((X - P[0])**2 + (Y - P[1])**2
            
        return g
            
    def getConstraintMin(self):
        return [self.Rd2] * (self.n_obs * 2)
        
    def getConstraintMax(self):
        return [np.inf] * (self.n_obs * 2)
        
    ############################################################
    # Bounds ###################################################
    ############################################################
    # * dimension check
    
    def getStateMin(self, k):
        state_min = [
            -np.inf,  # X
            -np.inf,  # Y
            self.init[2] - np.pi,  # Psi, constrained to one round for fast solving
            -2.5 / self.v_s,  # v
            -0.75 / self.delta_s,  # delta
        ]
        return state_min
    
    def getStateMax(self, k):
        state_max = [
            np.inf,  # X
            np.inf,  # Y 
            self.init[2] + np.pi,  # Psi, constrained to one round for fast solving
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