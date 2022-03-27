""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


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
            x0 = np.zeros((self.n_free,1))

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


class Referenceline_Opt_Free:
    def __init__(self, P_orig, 
                 type='opt',
                 interval=1.5,
                 a=1,
                 b=0,
                 g=0,
                 d=10,
                 dis_max=0.5):
        # P should be organized as [[x_0, y_0],[x_1,y_1],...]

        self.type = type  # optmized or spline

        self.interval = interval  # [m]

        # weighting coefficients
        self.alpha = a  # first derivative
        self.beta = b  # second derivative
        self.gamma = g  # square of kappa
        self.delta = d  # distance of points
        
        self.dis_max = dis_max  # maximum moving distance for x and y

        # initialization
        self.P_all = []  # all the points' initial location

        self.P_orig = np.asarray(P_orig) 
        self.n_orig = self.P_orig.shape[0]

        self.insert()

    def m_fx(self, index):
        """ Index recursion for fixed points """
        return index % self.n_orig

    def m(self, index):
        """ Index recursion for all points """
        return index % self.n_all

    def insert(self):
        """ Insert free points according to interval """
        for i in range(self.n_orig):
            # the first point is original point 
            self.P_all.append(self.P_orig[i,:].tolist())

            tmp_P_this = np.array(self.P_orig[self.m_fx(i)])
            tmp_P_next = np.array(self.P_orig[self.m_fx(i+1)])
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

            for j in range(int(num_insert)):
                self.P_all.append((tmp_P_this + (j+1) * vec_each).tolist())

        # convert to ndarrays in the end
        self.P_all = np.asarray(self.P_all)
        self.n_all  = self.P_all.shape[0]

        # convert to casadi format
        self.dx_opt = ca.SX.sym('dx', self.n_all)
        self.dy_opt = ca.SX.sym('dy', self.n_all)
        self.Z = ca.vertcat(self.dx_opt, self.dy_opt)
        # final construction
        self.x = self.P_all[:,0] + self.dx_opt
        self.y = self.P_all[:,1] + self.dy_opt

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
                    self.gamma * ca.power(self.kappa[i],2.0) + \
                    self.delta * ca.power(ca.sqrt(ca.power(dx[i], 2.0) + ca.power(dy[i], 2.0)) - self.interval, 2.0)

            # solve NLP
            nlp  = {'f': self.f, 'x': self.Z}
            opts = {"ipopt.tol": 1e-7}
            solver = ca.nlpsol("solver", "ipopt", nlp)
            print(solver)

            # initial guess as all 0
            x0 = np.zeros((self.n_all*2,1))

            # warm start solving
            lbx_warm = x0 - 0.01
            ubx_warm = x0 + 0.01
            sol_warm = solver(x0=x0, lbx=lbx_warm, ubx=ubx_warm)
            x        = np.asarray(sol_warm['x'])

            # real solving
            lbx = x0 - self.dis_max
            ubx = x0 + self.dis_max
            sol = solver(x0=x, lbx=lbx, ubx=ubx)

            # results
            self.Z_sol = sol['x']
            self.Z_sol = np.asarray(self.Z_sol)
            self.dx_sol = self.Z_sol[0:self.n_all]
            self.dy_sol = self.Z_sol[self.n_all:]
            print('Z_sol: ', self.Z_sol)
            self.P_all_sol = self.P_all + np.concatenate((self.dx_sol, self.dy_sol), axis=1)