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
    def __init__(self, X, Y, A, K, alpha=1):
        """
        Initialization.
        :param X: features (n, p)
        :param Y: crowd labels (n, M)
        :param A: instance assignment with 1 => assigned; 0 => not assigned (n, M)
        :param K: (K+1) is the total number of classes
        """
        self.n = X.shape[0]  # the number of instances
        self.p = X.shape[1]  # feature dimension
        self.M = Y.shape[1]  # the number of crowd annotators
        self.K = K
        self.alpha = alpha   # assignment prob
        # data preparation
        self.X = X                               # features (n, p)
        self.XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))  # X X^T (n, p, p)
        self.Y = Y                               # crowd labels (n, M)
        self.Y_onehot = self.compute_Y_onehot()  # crowd labels in one-hot form (n, K, M)
        self.A = A                               # binary assignment indicator (n, M)

    def compute_D(self, tmp_beta):
        K = self.K
        p = self.p
        tmp_beta = tmp_beta.reshape((p * K, 1))
        I_pK = np.identity(K * p)
        D = I_pK - tmp_beta @ np.transpose(tmp_beta)
        return D

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k + 1)).astype(int)
        return Y_onehot

    def compute_pikm(self, tmp_beta, tmp_sigma):
        """
        Compute exp( X_i beta_k / sigma_m ) / normalization
        :param tmp_beta: calculate at this beta value
        :param tmp_sigma: calculate at this sigma value
        :return:
        """
        M = self.M
        value_ik = (self.X @ np.transpose(tmp_beta)).reshape(self.n, self.K, 1)  # (n, K, 1)
        value_ikm = value_ik / tmp_sigma.reshape(1, 1, M)                        # (n, K, M)
        value_ikm[value_ikm > 100] = 100                                         # prevent Inf
        value_ikm[value_ikm < -100] = -100                                       # prevent -Inf
        value_ikm = np.exp(value_ikm)                                            # (n, K, M)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1                     # (n, M); +1 is due to class 0
        p_ikm = value_ikm / value_sum                                            # (n, K, M)
        return p_ikm                                                             # (n, K, M)

    def compute_A_diff(self, p_ikm):
        """
        Compute a_im { I(Y_im = k) - p_ikm } X_i / sigma_m
        :param p_ikm: exp( X_i beta_k / sigma_m ) / normalization
        :return:
        """
        diff = self.Y_onehot - p_ikm           # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def compute_derivative(self, tmp_beta, tmp_sigma):
        """
        Compute first- and second-order derivatives.
        :param tmp_beta: calculate at this beta value
        :param tmp_sigma: calculate at this sigma value
        :return:
        """
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        ##################################### 1st derivative #####################################
        # partial beta
        p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)  # (n, K, M)
        A_diff = self.compute_A_diff(p_ikm)
        delta = A_diff / tmp_sigma.reshape(1, 1, M)     # (n, K, M)
        delta = delta.sum(axis=2)                       # (n, K)
        partial_beta = np.transpose(delta) @ self.X     # (K, n) @ (n, p) = (K, p)

        # partial sigma
        partial_sigma = - A_diff / tmp_sigma.reshape(1, 1, M) ** 2             # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(tmp_beta)).reshape((n, K, 1))  # (n, K, M)
        partial_sigma = partial_sigma.sum(axis=(0, 1))                         # (M,)

        ##################################### 2st derivative #####################################
        A11 = np.zeros((K * p, K * p))        # partial beta^2:  (pK, pK)
        A22 = -2 * partial_sigma / tmp_sigma  # partial sigma^2: (M, 1)
        for j in range(K):
            for k in range(K):
                App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                    # (n, M)
                Sigma_jk = - App / (tmp_sigma.reshape(1, M) ** 2)     # (n, M)
                Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))             # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))  # (n, M, p, p)
                # A11
                A11[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
                # A22
                Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                for m in range(M):
                    A22[m] += tmp_beta[j] @ Sigma_jk[m] @ tmp_beta[k] / tmp_sigma[m] ** 2

        return partial_beta.ravel(), partial_sigma, A11, A22

