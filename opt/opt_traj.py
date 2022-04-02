""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


from . import calc

import time
import sys
import casadi as ca
import numpy as np
import yaml


class Trajectory_Opt:
    def __init__(self, ref) -> None:
        self.x = np.asarray(ref["x"])
        self.y = np.asarray(ref["y"])
        self.s = np.asarray(ref["s"])
        self.kappa = np.asarray(ref["kappa"])
        self.bl = np.asarray(ref["bl"])  # +
        self.br = np.asarray(ref["br"])  # -
        
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
        # System Definition ########################################
        ############################################################
        
        # number
        nx = 6  # of state
        nu = 2  # of control 
        
        # Declare decision variables
        # use scaling for decisions variables
        
        # scaled state vector
        vx_n    = ca.SX.sym('vx')
        vy_n    = ca.SX.sym('vy')
        dpsi_n  = ca.SX.sym('dpsi')
        n_n     = ca.SX.sym('n')
        chi_n   = ca.SX.sym('chi')
        delta_n = ca.SX.sym('delta')
        # the scaling for state vector (same as their maximum value)
        vx_s    = 20
        vy_s    = 1
        dpsi_s  = 2
        n_s     = 3
        chi_s   = 1
        delta_s = 0.5
        # state vector with normal scale
        vx    = vx_n * vx_s
        vy    = vy_n * vy_s
        dpsi  = dpsi_n * dpsi_s
        n     = n_n * n_s
        chi   = chi_n * chi_s
        delta = delta_n * delta_s
        # formulate array
        x   = ca.vertcat(vx_n, vy_n, dpsi_n, n_n, chi_n, delta_n)
        x_s = np.array([vx_s, vy_s, dpsi_s, n_s, chi_s, delta_s])
        
        # scaled input vector
        ddelta_n = ca.SX.sym('ddelta')
        T_n      = ca.SX.sym('T')
        # the scaling for input vector (same as their maximum value)
        ddelta_s = 2
        T_s      = 200
        # input vector with normal scale
        ddelta = ddelta_n * ddelta_s
        T      = T_n * T_s
        # formulate array
        u = ca.vertcat(ddelta_n, T_n)
        u_s = np.array([ddelta_s, T_s])
        
        # additional 
        kappa = ca.SX.sym('kappa')  # curvature
        
        
        # model parameters
        By = 9.882
        Cy = 1.111
        Dy = -2.559
        Ey = 0.2949
        
        R  = 0.2186
        m  = 230
        g  = 9.81
        Iz = 138.53
        lf = 0.858
        lr = 0.702
        l  = lr + lf
        
        Fzf = m * g * lr / l
        Fzr = m * g * lf / l
        
        # intermediate value
        # the torque also drives wheels
        Fx  = T / R * 0.94  # 94% due to acceleration of wheels
        # no need to multiply two because Fzf already twice
        Fyf = Fzf * By * Cy * Dy * (ca.atan((vy + dpsi * lf) / vx) - delta)
        Fyr = Fzr * By * Cy * Dy * (ca.atan((vy - dpsi * lr) / vx))
        
        # model descriptions
        # convert to s-based dynamics instead of t-based
        dt    = (1 - n * kappa) / (vx * ca.cos(chi) - vy * ca.sin(chi))  # 1/ds or dt/ds integrate to get t
        
        dvx   = dt * ((Fx - Fyf * ca.sin(delta)) / m + dpsi * vy)
        dvy   = dt * ((Fyf * ca.cos(delta) + Fyr) / m - dpsi * vx)
        ddpsi = dt * ((Fyf * lf * ca.cos(delta) + Fx * lf * ca.sin(delta) - Fyr * lr) / Iz)
        dn    = dt * (vx * ca.sin(chi) + vy * ca.cos(chi))
        dchi  = dt * dpsi - kappa
        # ddelta just ddelta

        # Model equations
        dx = ca.vertcat(dvx, dvy, ddpsi, dn, dchi, ddelta) / x_s

        # Objective term
        # L = dt
        L = 5 * dt + 2 * ddelta * ddelta + 0.00001 * T * T

        # Continuous time dynamics
        f = ca.Function('f', [x, u, kappa], [dx, L, dt], ['x', 'u', 'kappa'], ['dx', 'L', 'dt'])
        
        ############################################################
        # Constraints & Guesses ####################################
        ############################################################
        
        # state
        # state_min and state_max are in the loop
        state_guess = [5.0 / vx_s, 0.0, 0.0, (self.bl[0]+self.br[0])/2 / n_s, 0.0, 0.0]
        state_init  = [1.0 / vx_s, 0.0, 0.0, (self.bl[0]+self.br[0])/2 / n_s, 0.0, 0.0]
        # input
        input_min = [
            -1.414 / ddelta_s,  # ddelta
            -100 / T_s,  # T
        ]
        input_max = [
            1.414 / ddelta_s,  # ddelta
            100 / T_s,  # T
        ]
        input_guess = [0.0] * nu

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
        # lap time
        dt_opt = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', nx)
        w.append(Xk)
        # * dimension check
        lbw.append(state_init)  # equality constraint on init
        ubw.append(state_init)
        w0.append(state_init)
        x_opt.append(Xk * x_s)

        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), nu)
            w.append(Uk)
            # * dimension check
            lbw.append(input_min)
            ubw.append(input_max)
            w0.append(input_guess)

            # State at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym('X_'+str(k)+'_'+str(j), nx)
                Xc.append(Xkj)
                w.append(Xkj)
                lbw.append([-np.inf] * nx)
                ubw.append([np.inf] * nx)
                # * dimension check
                w0.append(state_guess)

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
                fj, qj, tj = f(Xc[j - 1], Uk, kappa_col)
                g.append(h[k] * fj - xp)
                lbg.append([0.0] * nx)  # equality constraints of collocation
                ubg.append([0.0] * nx)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                J += B[j] * qj * h[k]
                
                # Add contribution to lap time
                dt_int += B[j] * tj * h[k]

            # Time segament
            dt_opt.append(dt_int)
            
            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), nx)
            w.append(Xk)
            # * dimension check
            state_min = [
                0.0 / vx_s,  # vx
                -np.inf,  # vy 
                -np.inf,  # dpsi
                self.br[k] / n_s,  # n
                -np.inf,  # chi
                -0.43 / delta_s,  # delta
            ]
            state_max = [
                5.0 / vx_s,  # vx
                np.inf,  # vy 
                np.inf,  # dpsi
                self.bl[k] / n_s,  # n
                np.inf,  # chi
                0.43 / delta_s,  # delta
            ]
            lbw.append(state_min)
            ubw.append(state_max)
            w0.append(state_guess)
            x_opt.append(Xk * x_s)
            u_opt.append(Uk * u_s)

            # Add equality constraint
            g.append(Xk_end-Xk)  # compact form
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_opt = ca.horzcat(*x_opt)  # horzcat for 2D matrix
        u_opt = ca.horzcat(*u_opt)
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

        # solver options
        opts = {"expand": True,
                "verbose": True,
                "ipopt.max_iter": 2000,
                "ipopt.tol": 1e-7}

        prob = {'f': J, 'x': w, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', prob)

        # Function to get x and u trajectories from w
        trajectories = ca.Function('trajectories', [w], [x_opt, u_opt, dt_opt], 
                                   ['w'], ['x', 'u', 'dt_opt'])

        t_solve_start = time.perf_counter()

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        t_solve_end = time.perf_counter()
        
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
            sys.exit(1)
        
        x_opt, u_opt, dt_opt = trajectories(sol['x'])
        self.x_opt = x_opt.full() # to numpy array
        self.u_opt = u_opt.full() # to numpy array
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
        