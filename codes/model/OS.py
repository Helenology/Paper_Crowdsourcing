#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 14:44
# @Author  : Helenology
# @Site    :
# @File    : OS.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import norm


class OS:
    def __init__(self, X, Y, A, K, beta=None, sigma=None):
        """

        :param X:
        :param Y:
        :param A:
        :param K:
        :param beta:
        :param sigma: annotator's parameter sigma
        """
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        self.beta = beta
        self.sigma = sigma
        # data preparation
        self.X = X
        self.XXT = self.compute_XXT()  # (n, p, p)
        self.Y = Y
        self.Y_onehot = self.compute_Y_onehot()  # (n, K, M)
        self.A = A

    def compute_XXT(self):
        """Conpute $X_i X_i^\top$ for $1 \leq i \leq n$"""
        XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))
        return XXT

    def compute_likelihood(self):
        p_ikm = np.zeros((self.n, (self.K + 1), self.M))  # (n, (K+1), M)
        p_ikm[:, 1:] = self.compute_pikm()
        p_ikm[:, 0] = 1 - p_ikm[:, 1:].sum(axis=1)  #
        p_ikm += 1e-10
        p_ikm /= p_ikm.sum(axis=1, keepdims=True)
        Y_onehot = np.ones((self.n, (self.K + 1), self.M))
        for k in range(self.K):
            Y_onehot[:, k, :] = (self.Y == k).astype(int)
        likelihood = self.A.reshape(self.n, 1, self.M) * Y_onehot * np.log(p_ikm)
        likelihood = likelihood.sum() / self.n
        return likelihood

    def compute_pikm(self):
        value_ik = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma.reshape(1, 1, self.M)
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # +1 due to class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k + 1)).astype(int)
        return Y_onehot

    def compute_A_diff(self):
        self.p_ikm = self.compute_pikm()       # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def derivative_calcu(self):
        ##################################### 1st derivative #####################################
        # partial beta
        A_diff = self.compute_A_diff()
        delta = A_diff / self.sigma.reshape(1, 1, self.M)  # (n, K, M)
        delta = delta.sum(axis=2)                          # (n, K)
        partial_beta = np.transpose(delta) @ self.X        # (K, n) @ (n, p) = (K, p)

        # partial sigma
        partial_sigma = -A_diff / self.sigma.reshape(1, 1, self.M) ** 2  # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(self.beta)).reshape((self.n, self.K, 1))  # (n, K, 1)
        partial_sigma = partial_sigma.sum(axis=(0, 1))  # (M,)

        ##################################### 2st derivative #####################################
        A11 = np.zeros((self.K * self.p, self.K * self.p))  # partial beta^2: (pK, pK)
        A22 = -2 * partial_sigma / self.sigma               # partial sigma^2 (M, 1)
        for j in range(self.K):
            for k in range(self.K):
                App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                   # (n, M)
                Sigma_jk = -App / (self.sigma.reshape((1, self.M))**2)               # (n, M)
                Sigma_jk = Sigma_jk.reshape((self.n, self.M, 1, 1))                  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((self.n, 1, self.p, self.p))  # (n, M, p, p)
                # A11
                A11[(j * self.p):((j + 1) * self.p), (k * self.p):((k + 1) * self.p)] = Sigma_jk.sum(
                    axis=(0, 1))  # (p, p)
                # A22
                Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                for m in range(self.M):
                    A22[m] += self.beta[j] @ Sigma_jk[m] @ self.beta[k] / self.sigma[m] ** 2
        A22 = np.diag(A22)
        return partial_beta, partial_sigma, A11, A22

    def one_step_update(self, steps=1):
        gradient_beta, gradient_sigma, Hessian_beta, Hessian_sigma = self.derivative_calcu()
        gradient_beta /= self.n
        gradient_sigma /= self.n
        Hessian_beta /= self.n
        Hessian_sigma /= self.n
        # beta update
        gradient_beta = gradient_beta.ravel()
        beta = self.beta.ravel() - np.linalg.inv(Hessian_beta) @ gradient_beta
        # sigma update
        sigma = self.sigma - np.linalg.inv(Hessian_sigma) @ gradient_sigma
        return beta, sigma
