""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


from . import calc
# new model added here
from .model.bi_3dof_ddelta import Bi3dofddelta
from .model.b3dc_LT import B3dcLT
from .model.dual_7dof_noa import Dual7dofnoa

import time
import sys
import casadi as ca
import numpy as np
import yaml


class Trajectory_Opt:
    def __init__(self, ref, config, model_type, param=None, previous_data=None) -> None:
        # from ref
        self.x = np.asarray(ref["x"])
        self.y = np.asarray(ref["y"])
        self.s = np.asarray(ref["s"])
        self.kappa = np.asarray(ref["kappa"])
        self.bl = np.asarray(ref["bl"])  # +
        self.br = np.asarray(ref["br"])  # -
        
        # from config
        self.nx = config["nx"]
        self.nu = config["nu"]
        self.pc = config["pc"]
        self.max_iter = config["max_iter"]
        
        # new model added here
        if model_type == 'bi_3dof_ddelta':
            self.model = Bi3dofddelta(ref, config)
        elif model_type == 'b3dc_LT':
            self.model = B3dcLT(ref, config)
        elif model_type == 'dual_7dof_noa':
            self.model = Dual7dofnoa(ref, config, param, previous_data)
        
        
    def optimize(self):
        """ start optimizing mintime """
        
        t_formulation_start = time.perf_counter()
        
        ############################################################
        # Track Processing #########################################
        ############################################################
        
        # close track
        self.kappa = np.append(self.kappa, self.kappa[0])
        
        # step
        h = calc.XY2h(self.x, self.y)
        steps = [i for i in range(self.kappa.size)]
        N = steps[self.kappa.size - 1]
        
        # interpolation, so that we can use kappa at kappa_interp(1.5)
        kappa_interp = ca.interpolant('kappa_interp', 'linear', [steps], self.kappa)
        
        ############################################################
        # Direct Gauss-Legendrel Collacation #######################
        ############################################################
        
        # Degree of interpolating polynomial
        d = 3

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

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # For plotting x and u given w
        x_opt = []
        u_opt = []
        # dstate
        dx_opt = []
        # carinfo
        carinfo_opt = []
        # lap time
        dt_opt = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', self.nx)
        w.append(Xk)
        # * dimension check
        lbw.append(self.model.state_init)  # equality constraint on init
        ubw.append(self.model.state_init)
        w0.append(self.model.state_init)
        x_opt.append(Xk * self.model.x_s)

        # Formulate the NLP
        for k in range(N):
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
            dt_int = 0
            for j in range(1,d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(d): xp = xp + C[r + 1, j] * Xc[r]
                
                # interpolate kappa at the collocation point
                kappa_col = kappa_interp(k + tau[j])

                # Append collocation equations
                fj, qj, tj = self.model.f(Xc[j - 1], Uk, kappa_col)
                g.append(h[k] * fj - xp)
                lbg.append([0.0] * self.nx)  # equality constraints of collocation
                ubg.append([0.0] * self.nx)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Integration
                # Add contribution to quadrature function
                J += B[j] * qj * h[k]
                # Add contribution to lap time
                dt_int += B[j] * tj * h[k]

            # Time segament
            dt_opt.append(dt_int)
            
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), self.nx)
            w.append(Xk)
            lbw.append(self.model.getStateMin(k))
            ubw.append(self.model.getStateMax(k))
            w0.append(self.model.getStateGuess(k))
            x_opt.append(Xk * self.model.x_s)
            u_opt.append(Uk * self.model.u_s)
            dx_opt.append(self.model.f_d(Xk, Uk, kappa_interp(k)))
            carinfo_opt.append(self.model.f_carinfo(Xk, Uk))

            # Add equality constraint
            g.append(Xk_end-Xk)  # compact form
            lbg.append([0.0] * self.nx)
            ubg.append([0.0] * self.nx)
            # Check other constraint
            if self.pc == True:
                g.append(self.model.f_g(Xk, Uk))
                lbg.append(self.model.getConstraintMin())
                ubg.append(self.model.getConstraintMax())

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_opt = ca.horzcat(*x_opt)  # horzcat for 2D matrix
        u_opt = ca.horzcat(*u_opt)
        dx_opt = ca.horzcat(*dx_opt)
        carinfo_opt = ca.horzcat(*carinfo_opt)
        dt_opt = ca.vertcat(*dt_opt)  # vertcat for 1D vec
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
                "ipopt.max_iter": self.max_iter,
                "ipopt.tol": 1e-7,
        }

        prob = {'f': J, 'x': w, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob, opts)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_opt, u_opt, dx_opt, carinfo_opt, dt_opt], 
                                   ['w'], ['x', 'u', 'dx_opt', 'carinfo_opt', 'dt_opt'])

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        t_solve_end = time.perf_counter()
        
        # if solver.stats()['return_status'] != 'Solve_Succeeded':
        #     print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
        #     sys.exit(1)
        
        x_opt, u_opt, dx_opt, carinfo_opt, dt_opt = trajectories(sol['x'])
        self.x_opt = x_opt.full().T # to numpy array
        self.u_opt = np.vstack((np.array([0.0]*self.nu), u_opt.full().T)) # to numpy array
        self.dx_opt = np.vstack((np.array([0.0]*self.nx), dx_opt.full().T)) # to numpy array
        self.carinfo_opt = np.vstack((np.array([0.0]*carinfo_opt.full().T.shape[1]), carinfo_opt.full().T)) # to numpy array
        self.t_opt = np.hstack((0.0, np.cumsum(dt_opt)))
        
        ############################################################
        # Information ##############################################
        ############################################################
        
        print("")
        print("[INFO] Number of optimized raw points: %d" % N)
        print("[INFO] Number of optimized decision variables: %d" % w.shape[0])
        print("[RESULT] Laptime: %.3f s" % self.t_opt[-1])
        print("[TIME] Formulation takes: %.3f s" % (t_formulation_end - t_formulation_start))
        print("[TIME] Solving takes: %.3f s" % (t_solve_end - t_solve_start))
        