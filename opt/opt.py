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


class Referenceline_Opt:
    def __init__(self, P_fixed, 
                 type='opt',
                 interval=1.5,
                 a=1,
                 b=0,
                 g=0):
        # P_fixed should be organized as [[x_0, y_0],[x_1,y_1],...]

        self.type = type  # optmized or spline

        self.interval = interval  # [m]

        # weighting coefficients
        self.alpha = a  # first derivative
        self.beta = b  # second derivative
        self.gamma = g  # square of kappa

        # initialization
        self.P_all = []  # all the points
        self.P0 = []  # all the free points
        self.M = []  # manipulation matrix
        self.index_fixed = []
        self.index_free = []

        self.P_fixed = np.asarray(P_fixed) 
        self.n_fixed = self.P_fixed.shape[0]

        self.insert()

    def m_fx(self, index):
        """ Index recursion for fixed points """
        return index % self.n_fixed

    def m(self, index):
        """ Index recursion for all points """
        return index % self.n_all

    def insert(self):
        """ Insert free points according to interval """
        for i in range(self.n_fixed):
            # the first point is fixed 
            self.P_all.append(self.P_fixed[i,:].tolist())
            self.index_fixed.append(len(self.P_all)-1)

            tmp_P_this = np.array(self.P_fixed[self.m_fx(i)])
            tmp_P_next = np.array(self.P_fixed[self.m_fx(i+1)])
            # distance between points
            distance = np.linalg.norm(tmp_P_next - tmp_P_this)
            # number of points to insert equally spaced (gross approximation)
            if distance > self.interval:
                num_insert = round(distance / self.interval) - 1
            else:
                num_insert = 0

            # the vector of this inter-space
            vec = tmp_P_next - tmp_P_this
            vec_each = vec / (num_insert + 1)

            # normalize and rotate to get unit manipulation vector
            vec_unit = vec / np.linalg.norm(vec)
            vec_unit = np.array([[0,-1],[1,0]]).dot(vec_unit)

            for j in range(int(num_insert)):
                self.P_all.append((tmp_P_this + (j+1) * vec_each).tolist())
                self.index_free.append(len(self.P_all)-1)
                self.M.append(vec_unit.tolist())

        # convert to ndarrays in the end
        self.index_fixed = np.asarray(self.index_fixed)
        self.index_free = np.asarray(self.index_free)
        self.P_all = np.asarray(self.P_all)
        self.P0 = self.P_all[self.index_free,:]
        self.M = np.asarray(self.M)

        self.n_free = self.P0.shape[0]
        self.n_all  = self.P_all.shape[0]

        # convert to casadi format
        self.M_opt = ca.SX(self.M)
        self.P0_opt = ca.SX(self.P0)
        self.Z_opt = ca.SX.sym('Z', self.n_free)
        self.P_opt = self.P0_opt + self.M_opt * self.Z_opt
        # final construction
        self.P_all_opt = ca.SX(self.P_all)
        self.P_all_opt[self.index_free,:] = self.P_opt
        self.x = self.P_all_opt[:,0]
        self.y = self.P_all_opt[:,1]

    def optimize(self):
        """ Start optimization process """
        if self.type == 'opt':
            # curvature is the raw material
            dx = ca.SX.zeros(self.n_all)
            dy = ca.SX.zeros(self.n_all)
            ddx = ca.SX.zeros(self.n_all)
            ddy = ca.SX.zeros(self.n_all)
            for i in range(self.n_all):
                dx[i] = self.x[self.m(i+1)] - self.x[self.m(i)]
                dy[i] = self.y[self.m(i+1)] - self.y[self.m(i)]
            for i in range(self.n_all):
                ddx[i] = dx[self.m(i+1)] - dx[self.m(i)]
                ddy[i] = dy[self.m(i+1)] - dy[self.m(i)]
            self.kappa = (dx*ddy - dy*ddx) / ca.power((dx*dx + dy*dy),1.5)

            # first and second derivative of curvature
            dkappa = ca.SX.zeros(self.n_all)
            ddkappa = ca.SX.zeros(self.n_all)
            for i in range(self.n_all):
                dkappa[i] = (self.kappa[self.m(i+1)] - self.kappa[self.m(i)]) / \
                    ca.sqrt(dx[i]*dx[i] + dy[i]*dy[i])
            for i in range(self.n_all):
                ddkappa[i] = (dkappa[self.m(i+1)] - dkappa[self.m(i)]) / \
                    ca.sqrt(dx[i]*dx[i] + dy[i]*dy[i])

            # cost function
            self.f = 0
            for i in range(self.n_all):
                self.f += self.alpha * ca.power(dkappa[i],2.0) + \
                    self.beta * ca.power(ddkappa[i],2.0) + \
                    self.gamma * ca.power(self.kappa[i],2.0)

            # solve NLP
            nlp  = {'f': self.f, 'x': self.Z_opt}
            opts = {"ipopt.tol": 1e-7}
            solver = ca.nlpsol("solver", "ipopt", nlp)
            print(solver)

            # initial guess as all 0
            x0 = np.zeros([self.n_free,1])

            # warm start solving
            lbx_warm = x0 - 0.01
            ubx_warm = x0 + 0.01
            sol_warm = solver(x0=x0, lbx=lbx_warm, ubx=ubx_warm)
            x        = np.asarray(sol_warm['x'])

            # real solving
            lbx = x0 - 1.0
            ubx = x0 + 1.0
            sol = solver(x0=x, lbx=lbx, ubx=ubx)

            # results
            self.Z_sol = sol['x']
            self.Z_sol = np.asarray(self.Z_sol)
            print('Z_sol: ', self.Z_sol)
            self.P_sol = self.P0 + self.M * self.Z_sol
            self.P_all_sol = self.P_all
            self.P_all_sol[self.index_free] = self.P_sol


class Trajectory_Opt:
    def __init__(self, ref) -> None:
        self.x = np.asarray(ref["x"])
        self.y = np.asarray(ref["y"])
        self.s = np.asarray(ref["s"])
        self.kappa = np.asarray(ref["kappa"])
        
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
        
        # Declare model variables
        # the state vector
        vx   = ca.SX.sym('vx')
        vy   = ca.SX.sym('vy')
        dpsi = ca.SX.sym('dpsi')
        n     = ca.SX.sym('n')
        chi   = ca.SX.sym('chi')
        delta = ca.SX.sym('delta')
        x = ca.vertcat(vx, vy, dpsi, n, chi, delta)
        # the input vector
        ddelta = ca.SX.sym('ddelta')
        T      = ca.SX.sym('T')
        u = ca.vertcat(ddelta, T)
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
        dx = ca.vertcat(dvx, dvy, ddpsi, dn, dchi, ddelta)

        # Objective term
        # just dt

        # Continuous time dynamics
        f = ca.Function('f', [x, u, kappa], [dx, dt], ['x', 'u', 'kappa'], ['dx', 'dt'])
        
        ############################################################
        # Constraints & Guesses ####################################
        ############################################################
        
        # state
        state_min = [
            0.0,  # vx
            -np.inf,  # vy 
            -np.inf,  # dpsi
            0.0,  # n
            -np.inf,  # chi
            -0.43,  # delta
        ]
        state_max = [
            5.0,  # vx
            np.inf,  # vy 
            np.inf,  # dpsi
            4.0,  # n
            np.inf,  # chi
            0.43,  # delta
        ]
        state_guess = [5.0, 0.0, 0.0, 2.0, 0.0, 0.0]  # guess n from midpoint
        state_init  = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]
        # input
        input_min = [
            -1.414,  # ddelta
            -100,  # T
        ]
        input_max = [
            1.414,  # ddelta
            100,  # T
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
        x_plot = []
        u_plot = []

        # "Lift" initial conditions
        Xk = ca.MX.sym('X0', nx)
        w.append(Xk)
        # * dimension check
        lbw.append(state_init)  # equality constraint on init
        ubw.append(state_init)
        w0.append(state_init)
        x_plot.append(Xk)

        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = ca.MX.sym('U_' + str(k), nu)
            w.append(Uk)
            # * dimension check
            lbw.append(input_min)
            ubw.append(input_max)
            w0.append(input_guess)
            u_plot.append(Uk)

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
            for j in range(1,d + 1):
                # Expression for the state derivative at the collocation point
                xp = C[0, j] * Xk
                for r in range(d): xp = xp + C[r + 1, j] * Xc[r]
                
                # interpolate kappa at the collocation point
                kappa_col = kappa_interp(k + tau[j])

                # Append collocation equations
                fj, qj = f(Xc[j - 1], Uk, kappa_col)
                g.append(h[k] * fj - xp)
                lbg.append([0.0] * nx)  # equality constraints of collocation
                ubg.append([0.0] * nx)

                # Add contribution to the end state
                Xk_end = Xk_end + D[j] * Xc[j - 1]

                # Add contribution to quadrature function
                J = J + B[j] * qj * h[k]

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k+1), nx)
            w.append(Xk)
            # * dimension check
            lbw.append(state_min)
            ubw.append(state_max)
            w0.append(state_guess)
            x_plot.append(Xk)

            # Add equality constraint
            g.append(Xk_end-Xk)  # compact form
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)

        # Concatenate vectors
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)
        x_plot = ca.horzcat(*x_plot)
        u_plot = ca.horzcat(*u_plot)
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
        trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u'])

        t_solve_start = time.perf_counter()

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        t_solve_end = time.perf_counter()
        
        if solver.stats()['return_status'] != 'Solve_Succeeded':
            print('\033[91m' + 'ERROR: Optimization did not succeed!' + '\033[0m')
            sys.exit(1)
        
        x_opt, u_opt = trajectories(sol['x'])
        self.x_opt = x_opt.full() # to numpy array
        self.u_opt = u_opt.full() # to numpy array
        
        ############################################################
        # Information ##############################################
        ############################################################
        
        print("")
        print("[TIME] Formulation takes: %.3fs" % (t_formulation_end - t_formulation_start))
        print("[TIME] Solving takes: %.3fs" % (t_solve_end - t_solve_start))
        
