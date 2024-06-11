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

    def compute_likelihood(self):
        p_ikm = np.zeros((self.n, (self.K+1), self.M))  # (n, (K+1), M)
        p_ikm[:, 1:] = self.compute_pikm()
        p_ikm[:, 0] = 1 - p_ikm[:, 1:].sum(axis=1)      #
        p_ikm += 1e-10
        p_ikm /= p_ikm.sum(axis=1, keepdims=True)
        Y_onehot = np.ones((self.n, (self.K+1), self.M))
        for k in range(self.K):
            Y_onehot[:, k, :] = (self.Y == k).astype(int)
        likelihood = self.A.reshape(self.n, 1, self.M) * Y_onehot * np.log(p_ikm)
        likelihood = likelihood.sum() / self.n
        return likelihood

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

    def compute_pikm(self):
        value_ik = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # Here +1 is because class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    # def GA_alg(self, max_steps=100, epsilon=1e-5, eta=0.01, lbd=0.01):
    #     """Gradient ascending"""
    #     while True:
    #         self.steps += 1
    #         likelihood = self.compute_likelihood()
    #         self.likelihood_list.append(likelihood)
    #         # gradient
    #         self.gradient = self.derivative_calcu(order=1) / self.n
    #         # penalty for $\|B^\top B\| = 1$
    #         penalty = np.zeros_like(self.gradient)
    #         penalty[0:(self.p * self.K)] = -(lbd * 2) * self.beta.ravel()
    #         # update theta
    #         self.theta = self.theta + eta * self.gradient + penalty  # Maximum it because MLE
    #         self.beta, self.sigma = self.copy_beta_sigma()
    #         # calculate difference
    #         theta_diff = norm(eta * self.gradient + penalty)
    #         # print(f"[step {self.steps}] with likelihood: {likelihood:.6f}; theta diff: {theta_diff: .6f}")
    #         if (theta_diff < epsilon) or (self.steps > max_steps) or \
    #                 np.isnan(theta_diff) or (likelihood < self.likelihood_list[-2]):
    #             break
    #     self.beta, self.sigma = self.copy_beta_sigma()

    def NR_alg(self, max_steps=10, epsilon=1e-5, lbd=0.1):
        while True:
            self.steps += 1
            likelihood = self.compute_likelihood()
            self.likelihood_list.append(likelihood)
            # gradient
            self.gradient, self.Hessian = self.derivative_calcu(order=2)
            self.gradient /= self.n
            self.Hessian /= self.n
            # update beta
            beta_diff = - np.linalg.inv(self.Hessian) @ self.gradient
            beta_new = self.beta.ravel() + beta_diff
            self.beta = beta_new.reshape(self.K, self.p)
            diff_norm = norm(beta_diff)
            print(f"[Step {self.steps}] beta difference norm:{diff_norm:.5f}")

            # terminal condition
            if (diff_norm < epsilon) or (self.steps > max_steps) \
                    or (np.isnan(diff_norm)) or (likelihood < self.likelihood_list[-2]):
                break
        return self.beta.ravel()


    def compute_A_diff(self):
        self.p_ikm = self.compute_pikm()       # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def derivative_calcu(self, order=1):
        ##################################### 1st derivative #####################################
        # partial beta
        A_diff = self.compute_A_diff()
        delta = A_diff / self.sigma.reshape(1, 1, self.M)  # (n, K, M)
        delta = delta.sum(axis=2)  # (n, K)
        partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)

        # gradient
        gradient = partial_beta.ravel()
        if order == 1:
            return gradient

        ##################################### 2st derivative #####################################
        A11 = np.zeros((self.K * self.p, self.K * self.p))  # partial beta^2: (pK, pK)
        for j in range(self.K):
            for k in range(self.K):
                App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                   # (n, M)
                Sigma_jk = -App / (self.sigma.reshape((1, self.M))**2)               # (n, M)
                Sigma_jk = Sigma_jk.reshape((self.n, self.M, 1, 1))                  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((self.n, 1, self.p, self.p))  # (n, M, p, p)
                # A11
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)

        matrix = np.zeros((self.K * self.p + self.M, self.K * self.p + self.M))
        matrix[:(self.K * self.p), :(self.K * self.p)] = A11
        return gradient, A11


if __name__ == "__main__":
    n = 100
    p = 10
    M = 5
    K = 2
    X = np.ones((n, p))
    Y = np.ones((n, M)) * 2
    Y[0] = -1
    A = np.ones((n, M))
    A[0] = 0

    mle_model = MLE(X, Y, A, K)
    # mle_model.first_derivative_calcu()
    # mle_model.second_derivative_calcu()
    likelihood = mle_model.compute_likelihood()





