#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/11 14:59
# @Author  : Helenology
# @Site    : 
# @File    : ORACLE_sigma.py
# @Software: PyCharm


from model.BaseModel import BaseModel
import numpy as np
from numpy.linalg import norm
from copy import copy


class ORACLE_sigma(BaseModel):
    def __init__(self, X, Y, A, K, true_beta, sigma):
        """
        Initialization.
        :param X: features (n, p)
        :param Y: crowd labels (n, M)
        :param A: instance assignment with 1 => assigned; 0 => not assigned (n, M)
        :param K: (K+1) is the total number of classes
        :param true_beta: true values of beta
        :param sigma: an estimator for sigma
        """
        BaseModel.__init__(self, X, Y, A, K)
        # parameter initialization
        self.beta = copy(true_beta)  # true values of beta
        self.sigma = copy(sigma)     # an estimator for sigma
        # optimization initialization
        self.steps = 0
        self.update = 0

    def derivative_calcu(self, tmp_beta, tmp_sigma):
        """
        Reconstruct the function from BaseModel without gradient of beta.
        :param tmp_beta: calculate at this beta value
        :param tmp_sigma: calculate at this sigma value
        :return:
        """
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        ##################################### 1st derivative #####################################
        # partial sigma
        p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)                         # (n, K, M)
        A_diff = self.compute_A_diff(p_ikm)
        partial_sigma = -A_diff / tmp_sigma.reshape(1, 1, M) ** 2              # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(tmp_beta)).reshape((n, K, 1))  # (n, K, M)
        partial_sigma = partial_sigma.sum(axis=(0, 1))                         # (M,)
        ##################################### 2st derivative #####################################
        A22 = -2 * partial_sigma / tmp_sigma                                            # partial sigma^2: (M, 1)
        for j in range(K):
            for k in range(K):
                App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                      # (n, M)
                Sigma_jk = -App / (tmp_sigma.reshape(1, M) ** 2)                        # (n, M)
                Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))                               # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))                    # (n, M, p, p)
                Sigma_jk = Sigma_jk.sum(axis=0)                                         # (M, p, p)
                for m in range(M):
                    A22[m] += tmp_beta[j] @ Sigma_jk[m] @ tmp_beta[k] / tmp_sigma[m] ** 2
        return partial_sigma, A22

    def update_alg(self, max_steps=10, tol=1e-5, true_sigma=None, echo=True):
        while True:
            self.steps += 1
            # gradient & Hessian
            partial_sigma, A22 = self.derivative_calcu(self.beta, self.sigma)
            partial_sigma = -partial_sigma / self.n
            A22 = -A22 / self.n
            # update beta
            sigma_diff = - A22 ** (-1) * partial_sigma
            self.sigma += sigma_diff
            if echo:
                print(f"######## [Step {self.steps}] ########")
                print(f"norm(sigma): {norm(partial_sigma):.7f}")
                if true_sigma is not None:
                    print(f"RMSE(sigma): {norm(self.sigma - true_sigma):.7f}")
            # terminal condition
            if (norm(partial_sigma) < tol) or (self.steps >= max_steps) or (norm(sigma_diff) < tol):
                break
        return self.sigma

