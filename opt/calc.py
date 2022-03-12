""" 
Copyright (C) 2022:
- Zijun Guo <guozijuneasy@163.com>
All Rights Reserved.
"""


import numpy as np
from scipy import interpolate as si


class Calc:
    def __init__(self, P, 
                 vehicle={'wf':1.22, 'wr':1.18}):
        # P should be organized as [[x_0, y_0],[x_1,y_1],...] with numpy format
        
        # initialization
        self.n = P.shape[0]
        self.P = P
        self.x = P[:,0]
        self.y = P[:,1]

        # vehicle information
        self.vehicle = vehicle
    
    #################################
    # UTILITIES #####################
    #################################

    def m(self, index):
        """ Index recursion for all points """
        return index % self.n

    def Dis(self):
        """ Calculate distance between points """
        # return n-1 data
        dis = []
        for i in range(self.n-1):
            dis.append(np.linalg.norm(self.P[i+1,:] - self.P[i,:]))
        dis = np.asarray(dis)
        return dis

    ##############################
    # STAND-ALONE ################
    ##############################
    # which you can treat as a function

    def CentersLeft(left, right):
        """ Calculate the centers of left and right cones
        based on left cones """
        # ! This version is not recommended, just for illustration purpose
        centers = []
        # find the closest right point to left points
        for i in range(left.shape[0]):
            min_dist = 9999
            min_index = -1
            for j in range(right.shape[0]):
                dist = np.linalg.norm(left[i,:] - right[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            center_x=(left[i,0]+right[min_index,0])/2.0
            center_y=(left[i,1]+right[min_index,1])/2.0
            centers.append(center_x)
            centers.append(center_y)
        centers = np.asarray(centers).reshape(-1, 2)
        return centers

    def CentersMin(left, right):
        """ Calculate the centers of left and right cones
        forming minimum centerpoints """
        indexes_left_based = []
        indexes_right_based = []
        centers = []
        # we should not get the centers by [finding the closest right point to left points]
        # because when turning left, there are more right cones than left cones. 
        # when turning right, there are more left cones than right cones. 
        # This results in an [unreasonaly crowded centers when turning right].
        for i in range(left.shape[0]):
            min_dist = 9999
            min_index = -1
            for j in range(right.shape[0]):
                dist = np.linalg.norm(left[i,:] - right[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            indexes_left_based.append([i, min_index])  # left and right point's index
        indexes_left_based = np.asarray(indexes_left_based).reshape(-1, 2)
        for j in range(right.shape[0]):
            min_dist = 9999
            min_index = -1
            for i in range(left.shape[0]):
                dist = np.linalg.norm(left[i,:] - right[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            indexes_right_based.append([min_index, j])  # left and right point's index
        indexes_right_based = np.asarray(indexes_right_based).reshape(-1, 2)
        for index_left_based in indexes_left_based:
            for index_right_based in indexes_right_based:
                if index_left_based[0] == index_right_based[0] and \
                    index_left_based[1] == index_right_based[1]:
                    center_x=(left[index_left_based[0],0] + \
                        right[index_left_based[1],0])/2.0
                    center_y=(left[index_left_based[0],1] + \
                        right[index_left_based[1],1])/2.0
                    centers.append([center_x, center_y])
        centers = np.asarray(centers).reshape(-1, 2)
        return centers, indexes_left_based, indexes_right_based

    def CentersVote(left, right):
        """ Calculate the centers of left and right cones
        on the basis of voting strategies """
        centers = []
        # The linkage matrix stores the voting information
        # The linkage with the highest vote is more likely to be selected as the connection
        linkage = np.zeros([left.shape[0], right.shape[0]])
        for i in range(left.shape[0]):
            min_dist = 9999
            min_index = -1
            for j in range(right.shape[0]):
                dist = np.linalg.norm(left[i,:] - right[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            linkage[i, min_index] += 1
        for j in range(right.shape[0]):
            min_dist = 9999
            min_index = -1
            for i in range(left.shape[0]):
                dist = np.linalg.norm(left[i,:] - right[j,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            linkage[min_index, j] += 1
        # The aim is to eliminate the pattern of 
        # [[1 1]    [[1 0]    [[2 1]    [[2 0]    [[1 1]
        #  [0 1]]    [1 1]]    [0 1]]    [1 1]]    [1 1]]
        I = linkage.shape[0]
        J = linkage.shape[1]
        for i in range(I):
            for j in range(J):
                if linkage[i, j] == 1:
                    if ((linkage[(i-1)%I, j] == 1 or linkage[(i-1)%I, j] == 2) and linkage[i, (j+1)%J] == 1) or \
                        (linkage[(i+1)%I, j] == 1 and (linkage[i, (j-1)%J] == 1 or linkage[i, (j-1)%J] == 2)):
                        # the pattern that needs to be eliminated
                        row_type = False
                        column_type = False
                        if linkage[(i-1)%I, j] == 2:
                            row_type = True
                        elif linkage[i, (j-1)%J] == 2:
                            column_type = True
                        else:
                            # iterate row above
                            linkage_row_nonzero = linkage[(i-1)%I,:][linkage[(i-1)%I,:].nonzero()]
                            for element in linkage_row_nonzero:
                                if element == 2:
                                    row_type = True
                            # iterate column to the left
                            linkage_column_nonzero = linkage[:,(j-1)%J][linkage[:,(j-1)%J].nonzero()]
                            for element in linkage_column_nonzero:
                                if element == 2:
                                    column_type = True
                        if row_type == True:
                            # get the number of ones, insert to center
                            linkage_row_nonzero_current = linkage[i,:][linkage[i,:].nonzero()]
                            num = linkage_row_nonzero_current.shape[0]
                            num = int(num / 2.0)  # 2->1, 5->2
                            # tranport the one
                            if linkage[i, (j-1)%J] == 0 and linkage[(i+1)%I, (j+1)%J] == 0:
                                # type 1
                                linkage[i, (j+num)%J] = 2
                                linkage[i, j] = 0
                            if linkage[i, (j+1)%J] == 0 and column_type == False:
                                # type 2
                                linkage[i, (j-num)%J] = 2
                                linkage[i, j] = 0
                        if column_type == True:
                            # get the number of ones, insert to center
                            linkage_column_nonzero_current = linkage[:,j][linkage[:,j].nonzero()]
                            num = linkage_column_nonzero_current.shape[0]
                            num = int(num / 2.0)  # 2->1, 5->2
                            # tranport the one
                            if linkage[(i-1)%I, j] == 0 and linkage[(i+1)%I, (j+1)%J] == 0:
                                # type 3
                                linkage[(i+num)%I, j] = 2
                                linkage[i, j] = 0
                            if linkage[(i+1)%I, j] == 0 and row_type == False:
                                # type 4
                                linkage[(i-num)%I, j] = 2
                                linkage[i, j] = 0
                        print(num)
        for i in range(I):
            for j in range(J):
                if linkage[i, j] == 2:
                    center_x=(left[i,0]+right[j,0])/2.0
                    center_y=(left[i,1]+right[j,1])/2.0
                    centers.append(center_x)
                    centers.append(center_y)
        centers = np.asarray(centers).reshape(-1, 2)
        return centers, linkage

    ##############################
    # LENGTH #####################
    ##############################

    def Length(self):
        """ Calculate curve length (progress) of each point """
        dis = self.Dis()
        length = [0]
        length_cum = 0.0  # cumulated curve length
        for i in range(self.n-1):
            length_cum += dis[i]
            length.append(length_cum)
        length = np.asarray(length)
        return length
        
    #################################
    # CURVATURE #####################
    #################################

    def CurvO1F(self):
        """ Calculate curvature using simplest forward first order accurate approximation """
        # * generally recommended
        dx = []
        dy = []
        ddx = []
        ddy = []
        # calculation
        for i in range(self.n):
            dx.append(self.x[self.m(i+1)] - self.x[self.m(i)])
            dy.append(self.y[self.m(i+1)] - self.y[self.m(i)])
        for i in range(self.n):
            ddx.append(dx[self.m(i+1)] - dx[self.m(i)])
            ddy.append(dy[self.m(i+1)] - dy[self.m(i)])
        # convert to numpy
        dx = np.asarray(dx)
        dy = np.asarray(dy)
        ddx = np.asarray(ddx)
        ddy = np.asarray(ddy)
        # curvature
        return (dx*ddy - dy*ddx) / np.power((dx*dx + dy*dy),1.5)

    def CurvO1C(self):
        """ Calculate curvature using simplest central first order accurate approximation """
        dx = []
        dy = []
        ddx = []
        ddy = []
        # calculation
        for i in range(self.n):
            dx.append(self.x[self.m(i+1)] - self.x[self.m(i)])
            dy.append(self.y[self.m(i+1)] - self.y[self.m(i)])
        for i in range(self.n):
            # * difference here
            ddx.append(dx[self.m(i)] - dx[self.m(i-1)])
            ddy.append(dy[self.m(i)] - dy[self.m(i-1)])
        # convert to numpy
        dx = np.asarray(dx)
        dy = np.asarray(dy)
        ddx = np.asarray(ddx)
        ddy = np.asarray(ddy)
        # curvature
        return (dx*ddy - dy*ddx) / np.power((dx*dx + dy*dy),1.5)

    def CurvO2C(self):
        """ Calculate curvature using central second order accurate approximation """
        # h[i] is the distance between P[i] & P[i+1]
        h = []
        # here, d means derivative to curve length, e.g. d()/ds
        dx = []
        dy = []
        ddx = []
        ddy = []

        for i in range(self.n):
            h.append(np.linalg.norm(self.P[self.m(i+1),:] - self.P[self.m(i),:]))
        for i in range(self.n):
            dx.append(\
                ((self.x[self.m(i+1)] - self.x[self.m(i)]) * np.power(h[self.m(i-1)],2) + (self.x[self.m(i)] - self.x[self.m(i-1)]) * np.power(h[self.m(i)],2)) /\
                (h[self.m(i)] * h[self.m(i-1)] * (h[self.m(i)] + h[self.m(i-1)]))\
                )
            dy.append(\
                ((self.y[self.m(i+1)] - self.y[self.m(i)]) * np.power(h[self.m(i-1)],2) + (self.y[self.m(i)] - self.y[self.m(i-1)]) * np.power(h[self.m(i)],2)) /\
                (h[self.m(i)] * h[self.m(i-1)] * (h[self.m(i)] + h[self.m(i-1)]))\
                )
        for i in range(self.n):
            ddx.append(\
                ((dx[self.m(i+1)] - dx[self.m(i)]) * np.power(h[self.m(i-1)],2) + (dx[self.m(i)] - dx[self.m(i-1)]) * np.power(h[self.m(i)],2)) /\
                (h[self.m(i)] * h[self.m(i-1)] * (h[self.m(i)] + h[self.m(i-1)]))\
                )
            ddy.append(\
                ((dy[self.m(i+1)] - dy[self.m(i)]) * np.power(h[self.m(i-1)],2) + (dy[self.m(i)] - dy[self.m(i-1)]) * np.power(h[self.m(i)],2)) /\
                (h[self.m(i)] * h[self.m(i-1)] * (h[self.m(i)] + h[self.m(i-1)]))\
                )
        # convert to numpy
        dx = np.asarray(dx)
        dy = np.asarray(dy)
        ddx = np.asarray(ddx)
        ddy = np.asarray(ddy)
        # curvature
        return (dx*ddy - dy*ddx) / np.power((dx*dx + dy*dy),1.5)

    def CurvCircle(self):
        """ calculate curvature using circle approximation with three points """
        # * recommended for sparse points
        kappa = []
        for i in range(self.n):
            # calculate value
            B = self.P[i,:]
            C = self.P[self.m(i+1),:]
            A = self.P[self.m(i-1),:]

            a = np.linalg.norm(B-C)
            b = np.linalg.norm(C-A)
            c = np.linalg.norm(A-B)

            # ! The area could be zero when 3 points colinear
            s = (a + b + c) / 2.0
            K = np.sqrt(s * (s - a) * (s - b) * (s - c))

            R = a * b * c / (4.0 * K)

            # process sign
            if np.cross(B-A, C-B) > 0:
                kappa.append(1.0/R)
            else:
                kappa.append(- 1.0/R)
        kappa = np.asarray(kappa)
        return kappa

    ###########################
    # Theta #####################
    ###########################

    def ThetaAtan(self):
        """ Calculate theta [rad] by cross-detecting atan 
        atan is only valid for [-pi/2, pi/2]. Range-crossing should be considered """
        cross = 0.0
        theta_first = np.arctan((self.y[1] - self.y[0]) / (self.x[1] - self.x[0]))
        # theta directily from atan
        theta_raw = [theta_first]  
        # final theta
        theta = [theta_first]
        dx = []
        dy = []
        # calculation
        for i in range(self.n):
            dx.append(self.x[self.m(i+1)] - self.x[self.m(i)])
            dy.append(self.y[self.m(i+1)] - self.y[self.m(i)])
        for i in range(1,self.n):
            # get raw data
            if dx[i] != 0.0:
                theta_raw.append(np.arctan(dy[i] / dx[i]))
            elif dy[i] >= 0:
                theta_raw.append(np.pi / 2.0)
            else:
                theta_raw.append(- np.pi / 2.0)

            # process raw data
            if np.abs(theta_raw[i]) > np.pi / 4.0:
                # then there is possibility for crossing 
                if theta_raw[i] > 0 and theta_raw[i-1] < 0:
                    # from - to +, clockwise rotation
                    cross -= np.pi
                elif theta_raw[i] < 0 and theta_raw[i-1] > 0:
                    # from + to -, counter-clockwise rotation
                    cross += np.pi

            theta.append(theta_raw[i] + cross)
        theta = np.asarray(theta)
        return theta

    def ThetaInt(self):
        """ Calculate theta [rad] by integrating curvature to curve length 
        kappa = d (theta) / d (s) """
        # ! not recommended, use ThetaAtan instead
        theta = [np.arctan((self.y[1] - self.y[0]) / (self.x[1] - self.x[0]))]
        theta_cum = theta[0]
        dis = self.Dis()
        kappa = self.CurvO1F()
        for i in range(self.n-1):
            theta_cum += kappa[i] * dis[i]
            theta.append(theta_cum)
        theta = np.asarray(theta)
        return theta

    #################################
    # BOUNDARY ######################
    #################################

    def BoundLinear(self, left, right):
        """ 1. Find the function of left and right boundary to curve length s: B_l(s) & B_r(s) 
        and return the left and right boundary of each points
        The function here is linear approximation 
        2. Shrink the boundary according to vehicle front and rear track width: w_f & w_r
        after shrinking, the vehicle becomes a stick """
        print('- Processing Boundaries ...')
        w = max(self.vehicle['wf'], self.vehicle['wr'])
        length = self.Length()
        theta  = self.ThetaAtan()
        Bl_sparse = []  # left +
        Bl_len    = []
        Br_sparse = []  # right -
        Br_len    = []
        
        # find the closet center points of left and right cones
        for left_i in left:
            min_dist = 9999
            min_index = -1
            for i in range(self.n):
                dist = np.linalg.norm(left_i - self.P[i,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            Bl_sparse.append(min_dist)
            Bl_len.append(length[min_index])
        Bl_sparse = np.asarray(Bl_sparse)
        Bl_len    = np.asarray(Bl_len)
        for right_i in right:
            min_dist = 9999
            min_index = -1
            for i in range(self.n):
                dist = np.linalg.norm(right_i - self.P[i,:])
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            Br_sparse.append( - min_dist)
            Br_len.append(length[min_index])
        Br_sparse = np.asarray(Br_sparse)
        Br_len    = np.asarray(Br_len)

        # the curve length of closest center can be [340, 1, 3]
        # so must sort first before spline evaluation
        index_sort = np.argsort(Bl_len)
        Bl_sparse = Bl_sparse[index_sort]
        Bl_len    = Bl_len[index_sort]
        index_sort = np.argsort(Br_len)
        Br_sparse = Br_sparse[index_sort]
        Br_len    = Br_len[index_sort]

        # spline evaluation
        # 1. ensure that the start and the end of boundary meets
        # 2. the index of Bl & Br leads the index of length, should rearrange Bl & Br
        # left
        f = si.interp1d(np.append(Bl_len-Bl_len[0], length[-1]), 
                        np.append(Bl_sparse, Bl_sparse[-1]), 
                        kind='linear')
        lead = np.where(Bl_len[0] == length)[0][0]
        rearrange_index = np.append(np.arange(length.shape[0]-lead, length.shape[0]), 
                                    np.arange(0, length.shape[0]-lead))
        Bl = f(length)[rearrange_index] - 0.5 * w  # rearranging & shrinking
        # right
        f = si.interp1d(np.append(Br_len-Br_len[0], length[-1]), 
                          np.append(Br_sparse, Br_sparse[-1]), 
                          kind='linear')
        lead = np.where(Br_len[0] == length)[0][0]
        rearrange_index = np.append(np.arange(length.shape[0]-lead, length.shape[0]), 
                                    np.arange(0, length.shape[0]-lead))
        Br = f(length)[rearrange_index] + 0.5 * w  # rearranging & shrinking
        print('Left Boundary: ', Bl)
        print('Right Boundary: ', Br)

        # calculate absolute position
        Bl_x = self.x - Bl * np.sin(theta)
        Bl_y = self.y + Bl * np.cos(theta)
        Br_x = self.x - Br * np.sin(theta)
        Br_y = self.y + Br * np.cos(theta)
        Bl_pos = np.hstack((Bl_x.reshape(-1, 1), Bl_y.reshape(-1, 1)))
        Br_pos = np.hstack((Br_x.reshape(-1, 1), Br_y.reshape(-1, 1)))

        return Bl, Br, Bl_pos, Br_pos
