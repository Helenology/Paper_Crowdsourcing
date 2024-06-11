#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/19 18:12
# @Author  : Helenology
# @Site    :
# @File    : MLE.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import norm
import copy
from scipy.optimize import minimize
from scipy.optimize import SR1
from scipy.optimize import Bounds

class MLE_python:
    def __init__(self, X, Y, A, K, initial_theta=None):
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
        if initial_theta is None:
            self.theta = np.ones((self.K * self.p + self.M, )) * 2
            self.theta[:(self.K * self.p)] = np.random.randn(self.K * self.p)
            self.theta[:(self.K * self.p)] /= norm(self.theta[:(self.K * self.p)])
        else:
            self.theta = initial_theta
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0
        self.likelihood_list = []

    def objective(self, theta):
        p_ikm = np.zeros((self.n, (self.K+1), self.M))  # (n, (K+1), M)
        p_ikm[:, 1:] = self.compute_pikm(theta)
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

    def copy_beta_sigma(self, theta):
        beta = theta[0:(self.p * self.K)].reshape(self.K, self.p)
        sigma = theta[(self.p * self.K):]
        return beta, sigma

    def compute_pikm(self, theta):
        beta, sigma = self.copy_beta_sigma(theta)
        value_ik = self.X.dot(np.transpose(beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / sigma
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # Here +1 is because class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def cons(self, theta):
        beta = theta[0:(self.K * self.p)]
        return norm(beta) - 1

    def cons_jac(self, theta):
        beta = np.zeros_like(theta)
        beta[0:(self.K * self.p)] = theta[0:(self.K * self.p)]
        return 2 * beta

    def python_min(self):
        x0 = self.theta
        # bounds = [(-np.inf, np.inf) for i in range(self.p * self.K)] + [(0, np.inf) for i in range(self.M)]
        eq_cons = {'type': 'ineq',
                   'fun': self.cons,
                   'jac': self.cons_jac}
        res = minimize(self.objective, x0,
                       method='trust-constr',  #method='SLSQP',
                       jac=self.first_derivative_calcu,
                       # hess=SR1(),
                       # bounds=bounds,
                       constraints=[eq_cons],
                       options={'disp': True,
                                'verbose': 1})  #'ftol': 1e-9,
        return res.x

    def compute_A_diff(self, theta):
        self.p_ikm = self.compute_pikm(theta)       # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def first_derivative_calcu(self, theta):
        ##################################### 1st derivative #####################################
        beta, sigma = self.copy_beta_sigma(theta)
        # partial beta
        A_diff = self.compute_A_diff(theta)
        delta = A_diff / sigma.reshape(1, 1, self.M)  # (n, K, M)
        delta = delta.sum(axis=2)  # (n, K)
        partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)

        # partial sigma
        partial_sigma = -A_diff / sigma.reshape(1, 1, self.M) ** 2  # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(beta)).reshape((self.n, self.K, 1))  # (n, K, 1)
        partial_sigma = partial_sigma.sum(axis=(0, 1))  # (M,)

        # gradient
        gradient = np.zeros_like(self.theta)
        gradient[:(self.K * self.p)] = partial_beta.ravel()
        gradient[(self.K * self.p):] = partial_sigma
        return -gradient

    def second_derivative_calcu(self, theta):
        beta, sigma = self.copy_beta_sigma(theta)
        # partial beta partial beta^T
        A11 = np.zeros((self.K * self.p, self.K * self.p))
        for j in range(self.K):
            for k in range(self.K):
                delta1 = 0
                if j == k:
                    delta1 = self.A * self.p_ikm[:, j, :]
                delta2 = -self.A * self.p_ikm[:, j, :] * self.p_ikm[:, k, :]
                delta = -(delta1 + delta2)
                delta = delta / (sigma**2)
                delta = delta.sum(axis=1).reshape(self.n, 1, 1)
                delta = delta * self.XXT
                delta = delta.sum(axis=0)
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = delta

        # partial sigma partial sigma^T
        diff = self.diff
        A22 = np.zeros((self.M, self.M))
        delta = diff * self.A.reshape(self.n, 1, self.M)
        remain = 2 * self.X.dot(np.transpose(beta)).reshape(self.n, self.K, 1)
        remain = remain / (sigma ** 3)
        partial_sigma = (delta * remain).sum(axis=(0, 1))
        A22 += np.diag(partial_sigma)

        a = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1) / (self.sigma**2)
        a = a**2
        b = (self.A.reshape(self.n, 1, self.M) * self.p_ikm * a).sum(axis=(0, 1))
        A22 -= np.diag(b)

        tmp = np.zeros((self.n, self.M))
        for j in range(self.K):
            for k in range(self.K):
                tmp_value = self.A * self.p_ikm[:, j, :] * self.p_ikm[:, k, :]
                a = (self.X.dot(self.beta[j].reshape(self.p, 1)) / self.sigma**2)
                b = (self.X.dot(self.beta[k].reshape(self.p, 1)) / self.sigma**2)
                tmp += tmp_value * a * b
        tmp = tmp.sum(axis=0)
        A22 += np.diag(tmp)

        matrix = np.zeros((self.K * self.p + self.M, self.K * self.p + self.M))
        matrix[:(self.K * self.p), :(self.K * self.p)] = A11
        matrix[(self.K * self.p):, (self.K * self.p):] = A22
        return -matrix