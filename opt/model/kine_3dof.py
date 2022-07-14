""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
Kinematic bicycle model with 3 DOFs
* this model corresponds to the one stipulated in TPCAP (https://www.tpcap.net/#/rules)
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
        self.fl = np.sqrt(np.square(lw+lf)+np.square(w/2))  # front length for obstacle constraint
        self.rl = np.sqrt(np.square(lr)+np.square(w/2))   # rear length for obstacle constraint
        self.fa = np.arctan(w/2/(lw+lf))  # front angle for obstacle constraint
        self.ra = np.pi - np.arctan(w/2/(lr))  # rear angle for obstacle constraint
        
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
        L = 10 * (a * a + v * v * dPsi * dPsi) + \
            10 * delta * delta + \
            10 * (Psi - self.ref[2]) * (Psi - self.ref[2]) 

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
    # Obstacles ################################################
    ############################################################
    
    def f_g(self, Xk):
        g = []
        X = Xk[0]
        Y = Xk[1]
        Psi = Xk[2]
        
        for l2 in self.obs:
            # right vehicle body line segment
            l1 = [
                [X + self.fl * ca.cos(- self.fa + Psi), Y + self.fl * ca.sin(- self.fa + Psi)],
                [X + self.rl * ca.cos(- self.ra + Psi), Y + self.rl * ca.sin(- self.ra + Psi)],
            ]
            sign1 = ((l2[0][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[0][0] - l1[1][0]) * (l1[0][1] - l1[1][1])) * \
                ((l2[1][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[1][0] - l1[1][0]) * (l1[0][1] - l1[1][1]))
            sign2 = ((l1[0][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[0][0] - l2[1][0]) * (l2[0][1] - l2[1][1])) * \
                ((l1[1][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[1][0] - l2[1][0]) * (l2[0][1] - l2[1][1]))
            g.append(ca.atan2(sign1, sign2))
            
            # left vehicle body line segment
            l1 = [
                [X + self.fl * ca.cos(self.fa + Psi), Y + self.fl * ca.sin(self.fa + Psi)],
                [X + self.rl * ca.cos(self.ra + Psi), Y + self.rl * ca.sin(self.ra + Psi)],
            ]
            sign1 = ((l2[0][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[0][0] - l1[1][0]) * (l1[0][1] - l1[1][1])) * \
                ((l2[1][1] - l1[1][1]) * (l1[0][0] - l1[1][0]) - (l2[1][0] - l1[1][0]) * (l1[0][1] - l1[1][1]))
            sign2 = ((l1[0][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[0][0] - l2[1][0]) * (l2[0][1] - l2[1][1])) * \
                ((l1[1][1] - l2[1][1]) * (l2[0][0] - l2[1][0]) - (l1[1][0] - l2[1][0]) * (l2[0][1] - l2[1][1]))
            g.append(ca.atan2(sign1, sign2))
        return g
            
    def getConstraintMin(self):
        return [- np.pi / 2] * (self.n_obs * 2)
        
    def getConstraintMax(self):
        return [np.pi] * (self.n_obs * 2)
        
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