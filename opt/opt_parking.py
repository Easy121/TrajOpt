""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
The library for automatic parking
"""


from . import calc
# new model added here
from .model.kine_3dof_line import Kine3dofLine
from .model.kine_3dof import Kine3dof

import time
import sys
import casadi as ca
import numpy as np
import yaml


class Parking_Opt:
    def __init__(self, init, ref, obs) -> None:
        # obstacle description
        self.obs = obs
        
        """ Model 
        model is trivial in parking problem, the crux becomes constraint formulation """
        
        # self.model = Kine3dofLine(init, ref, obs)  # infeasible
        self.model = Kine3dof(init, ref, obs)
        
        # X, Y, Psi, v, delta
        self.nx = 5
        # a, ddelta
        self.nu = 2
        # init and ref
        self.init = list(np.array(self.model.init + [0.0, 0.0]) / self.model.x_s)
        self.ref = list(np.array(self.model.ref + [0.0, 0.0]) / self.model.x_s)
        
        self.N = self.model.N
        
        
    def optimize(self):
        """ start optimizing mintime """
        
        t_formulation_start = time.perf_counter()
        
        ############################################################
        # Direct Gauss-Legendrel Collacation #######################
        ############################################################
        
        # Degree of interpolating polynomial
        d = 1  # used to be 3 for complex model

        # Get collocation points
        tau = np.append(0, ca.collocation_points(d, 'legendre'))

        # Coefficients of the collocation equation
        C = np.zeros((d+1,d+1))

        # Coefficients of the continuity equation
        D = np.zeros(d+1)

        # Coefficients of the quadrature function
        B = np.zeros(d+1)

        # Construct polynomial basis
        for j in range(d+1):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(d+1):
                if r != j:
                    p *= np.poly1d([1, -tau[r]]) / (tau[j]-tau[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(d+1):
                C[j,r] = pder(tau[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        ############################################################
        # Nonlinear Programming Formulation ########################
        ############################################################

        t_total = ca.MX.sym('t_total', 1)
        dt = t_total / self.N

        # Start with t_total in
        w = [t_total]
        w0 = [[10]]  # TODO determine automatically
        lbw = [[1]]  # set minimum time to 1s to avoid 0s solution
        ubw = [[60]]  # set maximum time to 60s
        J = 100 * t_total
        g = []
        lbg = []
        ubg = []

        # For plotting x and u given w
        x_opt = []
        u_opt = []
        # dstate
        dx_opt = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', self.nx)
        X0 = Xk
        w.append(Xk)
        # * dimension check
        lbw.append(self.init)  # equality constraint on init
        ubw.append(self.init)
        w0.append(self.init)
        x_opt.append(Xk * self.model.x_s)

        # Formulate the NLP
        for k in range(self.N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), self.nu)
            w.append(Uk)
            # * dimension check
            lbw.append(self.model.getInputMin(k))
            ubw.append(self.model.getInputMax(k))
            w0.append(self.model.getInputGuess(k))

            # State at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), self.nx)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append([-np.inf] * self.nx)
                ubw.append([np.inf] * self.nx)
                # * dimension check
                w0.append(self.model.getStateGuess(k))

            # Loop over collocation points
            Xk_end = D[0] * Xk
            for j in range(1,d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(d): xp = xp + C[r + 1, j] * Xc[r]
                
                # Append collocation equations
                fj, qj = self.model.f(Xc[j - 1], Uk)
                g.append(dt * fj - xp)
                lbg.append([0.0] * self.nx)  # equality constraints of collocation
                ubg.append([0.0] * self.nx)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Integration
                # Add contribution to quadrature function
                J += B[j] * qj * dt

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.nx)
            w.append(Xk)
            lbw.append(self.model.getStateMin(k))
            ubw.append(self.model.getStateMax(k))
            w0.append(self.model.getStateGuess(k))
            x_opt.append(Xk * self.model.x_s)
            u_opt.append(Uk * self.model.u_s)
            dx_opt.append(self.model.f_d(Xk, Uk))

            # Add equality constraint
            g.append(Xk_end-Xk)  # compact form
            lbg.append([0.0] * self.nx)
            ubg.append([0.0] * self.nx)
            # DONE add constraints here
            if self.obs is not None:
                g = g + self.model.f_g(Xk)
                lbg.append(self.model.getConstraintMin())
                ubg.append(self.model.getConstraintMax())

        g.append(Xk)  # reference final pose
        lbg.append(self.ref)
        ubg.append(self.ref)

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_opt = ca.horzcat(*x_opt)  # horzcat for 2D matrix
        u_opt = ca.horzcat(*u_opt)
        dx_opt = ca.horzcat(*dx_opt)
        w0 = np.concatenate(w0)
        lbw = np.concatenate(lbw)
        ubw = np.concatenate(ubw)
        lbg = np.concatenate(lbg)
        ubg = np.concatenate(ubg)

        t_formulation_end = time.perf_counter()

        ############################################################
        # Create an NLP solver #####################################
        ############################################################
        
        t_solve_start = time.perf_counter()

        # solver options
        opts = {"expand": True,
                "verbose": True,
                "ipopt.max_iter": 600,
                "ipopt.tol": 1e-7,
                "ipopt.linear_solver": 'mumps',
                # "ipopt.linear_solver": 'ma57',
        }

        prob = {'f': J, 'x': w, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_opt, u_opt, dx_opt, t_total], 
                                   ['w'], ['x_opt', 'u_opt', 'dx_opt', 't_total'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        t_solve_end = time.perf_counter()
        
        # if solver.stats()['return_status'] != 'Solve_Succeeded':
        #     print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
        #     sys.exit(1)
        
        x_opt, u_opt, dx_opt, t_total = trajectories(sol['x'])
        # column: different states or inputs, row: timeline
        self.x_opt = x_opt.full().T # to numpy array, the first x_opt is idle but kept for completeness
        self.u_opt = np.vstack((np.array([0.0]*self.nu), u_opt.full().T)) # to numpy array
        self.dx_opt = np.vstack((np.array([0.0]*self.nx), dx_opt.full().T)) # to numpy array
        self.t_opt = np.linspace(0, t_total.full()[0][0], self.N + 1)
        
        ############################################################
        # Information ##############################################
        ############################################################
        
        print("")
        print("[INFO] Number of optimized raw points: %d" % self.N)
        print("[INFO] Number of optimized decision variables: %d" % w.shape[0])
        print("[TIME] Formulation takes: %.3f s" % (t_formulation_end - t_formulation_start))
        print("[TIME] Solving takes: %.3f s" % (t_solve_end - t_solve_start))
        print("[RESULT] Minimum parking time: %.3f s" % self.t_opt[-1])
        