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


class OS(BaseModel):
    def __init__(self, X, Y, A, K, beta, sigma):
        BaseModel.__init__(self, X, Y, A, K)
        # parameter initialization
        self.beta = beta
        self.sigma = sigma
        self.theta = np.zeros(self.K * self.p + self.M)
        self.theta[0:(self.K * self.p)] = self.beta.ravel()
        self.theta[(self.K * self.p):(self.K * self.p + self.M)] = self.sigma
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0
        self.update = 0
    def copy_beta_sigma(self):
        beta = self.theta[0:(self.p * self.K)].reshape(self.K, self.p)
        sigma = self.theta[(self.p * self.K):]
        return beta, sigma

    def update_alg(self, max_steps=1, tol=1e-5, true_beta=None):
        while True:
            self.steps += 1
            print(f"######## [Step {self.steps}] ########")
            # gradient & Hessian
            self.gradient, self.Hessian = self.derivative_calcu(self.beta, self.sigma)
            self.gradient = -self.gradient / self.n
            self.Hessian = -self.Hessian / self.n
            # update theta = (beta, sigma)
            theta_diff = -np.linalg.inv(self.Hessian) @ self.gradient
            self.theta = self.theta + theta_diff
            self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
            self.beta, self.sigma = self.copy_beta_sigma()
            print(f"norm(gradient): {norm(self.gradient):.7f}")
            if true_beta is not None:
                print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
            # terminal condition
            if (norm(self.gradient) < tol) or (self.steps >= max_steps) or (norm(theta_diff) < tol):
                break
        # self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
        self.beta, self.sigma = self.copy_beta_sigma()
        return self.beta.ravel(), self.sigma

    # def ALNR_alg(self, max_updates=1, max_steps=1, tol=1e-5, sig=0.01, lbd=0.01, rho=2, true_beta=None):
    #     np.random.seed(0)
    #     self.update = 0
    #     while self.update < max_updates:
    #         print(f"######## [Update {self.update}] ########")
    #         self.steps = 0
    #         self.gradient = np.Inf
    #         beta_diff = np.Inf
    #         while (norm(self.gradient) > tol) and (self.steps < max_steps) and (norm(beta_diff) > tol):
    #             print(f"######## [Step {self.steps}] ########")
    #             # gradient & Hessian
    #             self.gradient, self.Hessian = self.derivative_calcu(self.beta, self.sigma)
    #             self.gradient = -self.gradient / self.n
    #             self.Hessian = -self.Hessian / self.n
    #             # penalty for $\|B^\top B\| = 1$
    #             pen_grad = np.zeros_like(self.gradient)
    #             pen_grad[0:(self.p * self.K)] = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
    #             pen_hess = np.zeros_like(self.Hessian)
    #             for j in range(self.K * self.p):
    #                 for k in range(self.K * self.p):
    #                     if j == k:
    #                         pen_hess[j, k] = lbd + sig * (norm(self.beta) ** 2 - 1 + 2 * (self.beta.ravel()[j]) ** 2)
    #                     else:
    #                         pen_hess[j, k] = 2 * sig * (self.beta.ravel()[j]) * (self.beta.ravel()[k])
    #             # update theta
    #             theta_diff = -np.linalg.inv(self.Hessian + pen_hess) @ (self.gradient + pen_grad)
    #             self.theta = self.theta + theta_diff
    #             # self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
    #             self.beta, self.sigma = self.copy_beta_sigma()
    #             print(f"norm(gradient): {norm(self.gradient):.7f}")
    #             if true_beta is not None:
    #                 print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
    #             self.steps += 1
    #         eq_violance = abs((norm(self.beta) ** 2 - 1) * 0.5)
    #         if eq_violance <= tol:
    #             break
    #         # update lambda (Lagrangian Multiplier)
    #         lbd = lbd + sig * eq_violance
    #         sig = sig * rho
    #         self.update += 1
    #
    #     self.beta, self.sigma = self.copy_beta_sigma()
    #     return self.beta.ravel(), self.sigma

    def check(self, initial_beta, initial_sigma, true_beta, true_sigma):
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        OS_beta = self.beta.reshape(K, p)
        # OS_sigma = self.sigma.reshape(M, 1)
        true_beta = true_beta.reshape(K, p)
        true_sigma = true_sigma.reshape(M, 1)
        init_grad, init_Hess = self.derivative_calcu(initial_beta, initial_sigma)
        init_A11 = init_Hess[0:(K * p), 0:(K * p)] / (n * M)
        diff_mom = np.sqrt(n * M) * init_A11 @ (OS_beta - true_beta).reshape((K * p), 1)

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
            diff_son += 1 / true_sigma[m, 0] * Sigma_m @ Dstar @ invSigma_m @ mL_m
            # diff_son += 1 / n * mL_m * true_beta.reshape(1, K * p) @ invSigma_m @ mL_m
        diff_son = - 1 / np.sqrt(n * M) * diff_son
        var1 = true_beta.reshape(1, K * p) @ Sigma_m @ true_beta.reshape(K * p, 1)
        var2 = true_beta.reshape(1, K * p) @ invSigma_m @ true_beta.reshape(K * p, 1)
        var = var1 * var2
        return diff_mom, diff_son, var

