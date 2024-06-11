#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 23:58
# @Author  : Helenology
# @Site    : 
# @File    : ORACLE_python.py
# @Software: PyCharm


import numpy as np
from numpy.linalg import norm
import copy
from scipy.optimize import minimize


class ORACLE:
    def __init__(self, X, Y, A, K, initial_beta, sigma):
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        # data preparation
        self.X = X
        self.XXT = self.compute_XXT()
        self.Y = Y
        self.Y_onehot = self.compute_Y_onehot()  # (n, K, M)
        self.A = A
        # parameter initialization
        self.beta = initial_beta
        self.sigma = sigma
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0
        self.likelihood_list = [-np.Inf]

    def objective(self, beta):
        p_ikm = np.zeros((self.n, (self.K+1), self.M))  # (n, (K+1), M)
        p_ikm[:, 1:] = self.compute_pikm(beta)
        p_ikm[:, 0] = 1 - p_ikm[:, 1:].sum(axis=1)
        p_ikm += 1e-5
        p_ikm /= p_ikm.sum(axis=1, keepdims=True)
        Y_onehot = np.ones((self.n, (self.K+1), self.M))
        for k in range(self.K):
            Y_onehot[:, k, :] = (self.Y == k).astype(int)
        likelihood = self.A.reshape(self.n, 1, self.M) * Y_onehot * np.log(p_ikm)
        likelihood = likelihood.sum() / self.n
        return -likelihood

    def compute_XXT(self):
        """Conpute $X_i X_i^\top$ for $1 \leq i \leq n$"""
        XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))
        return XXT

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))                      # here we neglect class 0
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k+1)).astype(int)  # here we neglect class 0
        return Y_onehot

    def compute_pikm(self, beta):
        beta = beta.reshape(self.K, self.p)
        value_ik = self.X.dot(np.transpose(beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # Here +1 is because class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def compute_A_diff(self, beta):
        self.p_ikm = self.compute_pikm(beta)   # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def cons(self, beta):
        return norm(beta) - 1

    def cons_jac(self, beta):
        return 2 * beta

    def python_min(self):
        x0 = self.beta.ravel()
        # bounds = [(-np.inf, np.inf) for i in range(self.p * self.K)] + [(0, np.inf) for i in range(self.M)]
        eq_cons = {'type': 'ineq',
                   'fun': self.cons,
                   'jac': self.cons_jac}
        res = minimize(self.objective, x0,
                       method='trust-constr',  #method='SLSQP',
                       jac=self.first_derivative_calcu,
                       hess=self.second_derivative_calcu,
                       constraints=[eq_cons],
                       options={'disp': True,
                                'verbose': 1})  #'ftol': 1e-9,
        return res.x

    def first_derivative_calcu(self, beta):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
        ##################################### 1st derivative #####################################
        # partial beta
        A_diff = self.compute_A_diff(beta)
        delta = A_diff / self.sigma.reshape(1, 1, self.M)  # (n, K, M)
        delta = delta.sum(axis=2)  # (n, K)
        partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)
        partial_beta = partial_beta.ravel()
        return -partial_beta / self.n

    def second_derivative_calcu(self, beta):
        # self.p_ikm = self.compute_pikm(beta)
        # ##################################### 2st derivative #####################################
        A11 = np.zeros((self.K * self.p, self.K * self.p))  # partial beta^2: (pK, pK)
        for j in range(self.K):
            for k in range(self.K):
                App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                   # (n, M)
                Sigma_jk = -App / (self.sigma.reshape((1, self.M))**2)               # (n, M)
                Sigma_jk = Sigma_jk.reshape((self.n, self.M, 1, 1))                  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((self.n, 1, self.p, self.p))  # (n, M, p, p)
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)

        return -A11 / self.n

