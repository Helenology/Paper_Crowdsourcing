#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 22:37
# @Author  : Helenology
# @Site    : 
# @File    : OS.py
# @Software: PyCharm

from model.BaseModel import BaseModel
import numpy as np
from numpy.linalg import norm
from copy import copy


class OS(BaseModel):
    def __init__(self, X, Y, A, K, alpha, beta, sigma):
        BaseModel.__init__(self, X, Y, A, K, alpha)
        # parameter initialization
        self.beta = copy(beta)
        self.sigma = copy(sigma)
        # optimization initialization
        self.steps = 0
        self.update = 0

    def update_alg(self, max_steps=1, tol=1e-5, true_beta=None, echo=True):
        while True:
            self.steps += 1
            # gradient & Hessian
            partial_beta, partial_sigma, A11, A22 = self.compute_derivative(self.beta, self.sigma)
            partial_beta /= - self.n
            partial_sigma /= - self.n
            A11 /= - self.n
            A22 /= - self.n
            # update beta and sigma
            beta_diff = -np.linalg.inv(A11) @ partial_beta
            self.beta += beta_diff.reshape(self.K, self.p)
            self.beta /= norm(self.beta)
            sigma_diff = - A22 ** (-1) * partial_sigma
            self.sigma += sigma_diff
            # gradient
            gradient = np.sqrt(norm(partial_beta) ** 2 + norm(partial_sigma) ** 2)
            if echo:
                print(f"######## [Step {self.steps}] ########")
                print(f"norm(gradient): {gradient:.7f}")
                if true_beta is not None:
                    print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
            # terminal condition
            if (norm(gradient) < tol) or (self.steps >= max_steps) or (norm(beta_diff) < tol):
                break
        return self.beta.ravel(), self.sigma

    def check(self, initial_beta, initial_sigma, true_beta, true_sigma):
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        OS_beta = self.beta.reshape(K, p)
        # OS_sigma = self.sigma.reshape(M, 1)
        true_beta = true_beta.reshape(K, p)
        true_sigma = true_sigma.reshape(M, 1)
        init_grad, init_Hess = self.compute_derivative(initial_beta, initial_sigma)
        init_A11 = init_Hess[0:(K * p), 0:(K * p)] / (n * M)
        diff_mom = np.sqrt(n * M) * init_A11 @ (OS_beta - true_beta).reshape((K * p), 1)

        diff_son = 0
        Dstar = self.compute_D(true_beta) # np.identity(K*p) - true_beta.reshape(K * p, 1) @ true_beta.reshape(1, K * p)
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
            diff_son += 1 / true_sigma[m, 0] * Sigma_m @ Dstar @ invSigma_m @ mL_m
            # diff_son += 1 / n * mL_m * true_beta.reshape(1, K * p) @ invSigma_m @ mL_m
        diff_son = - 1 / np.sqrt(n * M) * diff_son
        var1 = true_beta.reshape(1, K * p) @ Sigma_m @ true_beta.reshape(K * p, 1)
        var2 = true_beta.reshape(1, K * p) @ invSigma_m @ true_beta.reshape(K * p, 1)
        var = var1 * var2
        return diff_mom, diff_son, var

