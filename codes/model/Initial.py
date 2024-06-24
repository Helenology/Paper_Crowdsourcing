#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 20:47
# @Author  : Helenology
# @Site    : 
# @File    : Initial.py
# @Software: PyCharm

from model.BaseModel import BaseModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from numpy.linalg import norm


class Initial(BaseModel):
    def __init__(self, X, Y, A, K):
        BaseModel.__init__(self, X, Y, A, K)

    def init_param(self):
        initial_b = np.zeros((self.M, self.K + 1, self.p))
        initial_beta = np.zeros((self.K + 1, self.p))
        initial_sigma = np.zeros(self.M)
        for m in range(self.M):
            y_m = self.Y[:, m]
            idx = (self.A[:, m] != 0)
            X_m = self.X[idx]
            y_m = y_m[idx]
            clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_m, y_m)
            initial_b[m] = clf.coef_
            initial_b[m] -= initial_b[m, 0]
            initial_sigma[m] = 1 / norm(initial_b[m])
            initial_beta += initial_b[m] * initial_sigma[m] / self.M
        initial_b = initial_b[:, 1:, :]  # (M, K, p)
        initial_beta = initial_beta[1:]
        initial_beta /= norm(initial_beta)  # normalize
        return initial_beta, initial_sigma, initial_b

    def check(self, init_beta, init_sigma, true_beta, true_sigma):
        """check under alpha=1"""
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        true_beta = true_beta.reshape(K, p)
        true_sigma = true_sigma.reshape(M)
        diff_mom = (init_beta - true_beta).ravel()

        diff_son = 0
        Dstar = np.identity(K * p) - true_beta.reshape(K * p, 1) @ true_beta.reshape(1, K * p)
        p_ikm = self.compute_pikm(true_beta, true_sigma)  # (n, K, M)
        A_diff = self.compute_A_diff(p_ikm)               # (n, K, M)
        for m in range(M):
            Sigma_m = np.zeros((K * p, K * p))                                                        # (pK, pK)
            for j in range(K):
                for k in range(K):
                    App = int(j == k) * (p_ikm[:, j, m]) - p_ikm[:, j, m] * p_ikm[:, k, m]            # (n, )
                    App = (self.A[:, m] * App).reshape((n, 1, 1))                                     # (n, 1, 1)
                    Sigma_jk = App * self.XXT                                                         # (n, p, p)
                    Sigma_m[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=0) / n  # (p, p)
            invSigma_m = np.linalg.inv(Sigma_m)
            mL_m = np.transpose(A_diff[:, :, m]) @ self.X  # (K, n) @ (n, p) = (K, p)
            mL_m = mL_m.reshape(K * p, 1)
            diff_son += 1 / (n * M) * true_sigma[m] * Dstar @ invSigma_m @ mL_m

        return diff_mom, diff_son
