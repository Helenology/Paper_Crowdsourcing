#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/19 11:42
# @Author  : Helenology
# @Site    : 
# @File    : ORACLE.py
# @Software: PyCharm

from model.BaseModel import BaseModel
import numpy as np
from numpy.linalg import norm


class ORACLE(BaseModel):
    def __init__(self, X, Y, A, K, beta, sigma):
        BaseModel.__init__(self, X, Y, A, K)
        # parameter initialization
        self.beta = beta    # initial estimator for beta
        self.sigma = sigma  # true parameter sigma
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0
        self.update = 0

    def derivative_calcu(self, tmp_beta, tmp_sigma):
        """
        Reconstruct of the function from BaseModel without gradient of sigma.
        :param tmp_beta:
        :param tmp_sigma:
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
        partial_beta = partial_beta.ravel()

        ##################################### 2st derivative #####################################
        A11 = np.zeros((K * p, K * p))                                                         # (pK, pK)
        for j in range(K):
            for k in range(K):
                App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]         # (n, M)
                App = self.A * App                                                             # (n, M)
                Sigma_jk = -App / (tmp_sigma.reshape(1, M) ** 2)                               # (n, M)
                Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))                                      # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))                           # (n, M, p, p)
                # A11
                A11[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)

        return partial_beta, A11

    def update_alg(self, max_steps=10, tol=1e-5, true_beta=None):
        while True:
            self.steps += 1
            print(f"######## [Step {self.steps}] ########")
            # gradient & Hessian
            self.gradient, self.Hessian = self.derivative_calcu(self.beta, self.sigma)
            self.gradient = -self.gradient / self.n
            self.Hessian = -self.Hessian / self.n
            # update beta
            beta_diff = -np.linalg.inv(self.Hessian) @ self.gradient
            self.beta = self.beta + beta_diff.reshape(self.K, self.p)
            print(f"norm(gradient): {norm(self.gradient):.7f}")
            if true_beta is not None:
                print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
            # terminal condition
            if (norm(self.gradient) < tol) or (self.steps >= max_steps) or (norm(beta_diff) < tol):
                break
        return self.beta.ravel() / norm(self.beta)

    # def NR_alg(self, max_steps=10, tol=1e-5, sig=0.01, lbd=0.001, rho=2, true_beta=None):
    #     self.update = 1
    #     while self.update < max_steps:
    #         print(f"######## [Update {self.update}] ########")
    #         self.steps = 0
    #         self.gradient = np.Inf
    #         beta_diff = np.Inf
    #         while (norm(self.gradient) > tol) and (self.steps < max_steps) and (norm(beta_diff) > tol):
    #             print(f"######## [Update {self.update}'s Step {self.steps}] ########")
    #             # gradient & Hessian
    #             self.gradient, self.Hessian = self.derivative_calcu(self.beta, self.sigma)
    #             self.gradient = -self.gradient / self.n
    #             self.Hessian = -self.Hessian / self.n
    #             # penalty for $\|B^\top B\| = 1$
    #             pen_grad = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
    #             pen_hess = np.zeros_like(self.Hessian)
    #             for j in range(self.K * self.p):
    #                 for k in range(self.K * self.p):
    #                     if j == k:
    #                         pen_hess[j, k] = lbd + sig * (norm(self.beta) ** 2 - 1 + 2 * (self.beta.ravel()[j]) ** 2)
    #                     else:
    #                         pen_hess[j, k] = 2 * sig * (self.beta.ravel()[j]) * (self.beta.ravel()[k])
    #             # update theta
    #             beta_diff = -np.linalg.inv(self.Hessian + pen_hess) @ (self.gradient + pen_grad)
    #             self.beta = self.beta + beta_diff.reshape(self.K, self.p)
    #             print(f"norm(gradient): {norm(self.gradient):.7f}")
    #             if true_beta is not None:
    #                 print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
    #             self.steps += 1
    #         eq_violance = (norm(self.beta) ** 2 - 1) * 0.5
    #         if eq_violance <= tol:
    #             break
    #         # update lambda (Lagrangian Multiplier)
    #         lbd = lbd + sig * eq_violance
    #         sig = sig * rho
    #         self.update += 1
    #
    #     return self.beta.ravel() / norm(self.beta)

    def check(self, true_beta, true_sigma):
        """check under alpha=1"""
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        ORA_beta = self.beta.ravel()
        true_beta = true_beta.reshape(K, p)
        true_sigma = true_sigma.reshape(M)
        diff_mom = ORA_beta.ravel() - true_beta.ravel()

        diff_son = 0
        p_ikm = self.compute_pikm(true_beta, true_sigma)  # (n, K, M)
        A_diff = self.compute_A_diff(p_ikm)               # (n, K, M)
        L1 = A_diff / true_sigma.reshape(1, 1, M)         # (n, K, M)
        L1 = L1.sum(axis=2)                               # (n, K)
        L1 = np.transpose(L1) @ self.X                    # (K, n) @ (n, p) -> (K, p)
        L1 = L1.reshape(K * p, 1)                         # (K * p, 1)
        L2 = 0
        for m in range(M):
            Sigma_m = np.zeros((K * p, K * p))                                                        # (pK, pK)
            for j in range(K):
                for k in range(K):
                    App = int(j == k) * (p_ikm[:, j, m]) - p_ikm[:, j, m] * p_ikm[:, k, m]            # (n, )
                    App = (self.A[:, m] * App).reshape((n, 1, 1))                                     # (n, 1, 1)
                    Sigma_jk = App * self.XXT                                                         # (n, p, p)
                    Sigma_m[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=0)      # (p, p)
            Sigma_m /= (true_sigma[m] ** 2)
            L2 += Sigma_m
        diff_son = np.linalg.inv(L2) @ L1
        return diff_mom, diff_son

