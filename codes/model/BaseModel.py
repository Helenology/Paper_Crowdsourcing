#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 20:46
# @Author  : Helenology
# @Site    : 
# @File    : BaseModel.py
# @Software: PyCharm


import numpy as np
from numpy.linalg import norm
import copy


class BaseModel:
    def __init__(self, X, Y, A, K):
        """

        :param X:
        :param Y:
        :param A:
        :param K:
        """
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        # data preparation
        self.X = X  # (n, p)
        self.XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))  # (n, p, p)
        self.Y = Y  # (n, M)
        self.Y_onehot = self.compute_Y_onehot()  # (n, K, M)
        self.A = A  # (n, M)
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k + 1)).astype(int)
        return Y_onehot

    def compute_pikm(self, tmp_beta, tmp_sigma):
        """

        :param tmp_beta: (K, p)
        :param tmp_sigma: (M, 1)
        :return:
        """
        M = tmp_sigma.shape[0]
        value_ik = (self.X @ np.transpose(tmp_beta)).reshape(self.n, self.K, 1)  # (n, K, 1)
        value_ikm = value_ik / tmp_sigma.reshape(1, 1, M)  # (n, K, M)
        value_ikm = np.exp(value_ikm)  # (n, K, M)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # (n, M); +1 due to class 0
        p_ikm = value_ikm / value_sum  # (n, K, M)
        return p_ikm  # (n, K, M)

    def compute_A_diff(self, p_ikm):
        diff = self.Y_onehot - p_ikm  # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff  # (n, K, M)
        return A_diff  # (n, K, M)

    def derivative_calcu(self, tmp_beta, tmp_sigma):  # For OS and MLE
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        ##################################### 1st derivative #####################################
        # partial beta
        p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)  # (n, K, M)
        A_diff = self.compute_A_diff(p_ikm)
        delta = A_diff / tmp_sigma.reshape(1, 1, M)  # (n, K, M)
        delta = delta.sum(axis=2)  # (n, K)
        partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)

        # partial sigma
        partial_sigma = -A_diff / tmp_sigma.reshape(1, 1, M) ** 2  # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(tmp_beta)).reshape((n, K, 1))  # (n, K, M)
        partial_sigma = partial_sigma.sum(axis=(0, 1))  # (M,)

        # gradient
        gradient = np.zeros(K * p + M)
        gradient[:(K * p)] = partial_beta.ravel()
        gradient[(K * p):] = partial_sigma

        ##################################### 2st derivative #####################################
        A11 = np.zeros((K * p, K * p))  # partial beta^2: (pK, pK)
        A22 = -2 * partial_sigma / tmp_sigma  # partial sigma^2 (M, 1)
        for j in range(K):
            for k in range(K):
                App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]  # (n, M)
                App = self.A * App  # (n, M)
                Sigma_jk = -App / (tmp_sigma.reshape(1, M) ** 2)  # (n, M)
                Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))  # (n, M, p, p)
                # A11
                A11[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
                # A22 & A12
                Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                for m in range(M):
                    A22[m] += tmp_beta[j] @ Sigma_jk[m] @ tmp_beta[k] / tmp_sigma[m] ** 2

        matrix = np.zeros((K * p + M, K * p + M))
        matrix[0:(K * p), 0:(K * p)] = A11
        matrix[(K * p):(K * p + M), (K * p):(K * p + M)] = np.diag(A22)
        return gradient, matrix

