#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/19 18:12
# @Author  : Helenology
# @Site    : 
# @File    : _MLE.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import norm
import copy
from scipy.optimize import minimize


class MLE:
    def __init__(self, X, Y, A, K, initial_beta, initial_sigma):
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
        self.beta = initial_beta                           # (K, p)
        self.sigma = initial_sigma                         # (M,)
        self.theta = np.ones((self.K * self.p + self.M,))  # (pK+M,)
        self.theta[0:(self.K * self.p)] = self.beta.ravel()
        self.theta[(self.K * self.p):] = self.sigma
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 1
        self.likelihood_list = [-np.Inf]

    def compute_XXT(self):
        """Compute $X_i X_i^\top$ for $1 \leq i \leq n$"""
        XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))
        return XXT

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

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k + 1)).astype(int)
        return Y_onehot

    def copy_beta_sigma(self):
        beta = self.theta[0:(self.p * self.K)].reshape(self.K, self.p)
        sigma = self.theta[(self.p * self.K):]
        return beta, sigma

    def compute_pikm(self):
        value_ik = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma.reshape(1, 1, self.M)
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # +1 due to class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def GA_alg(self, max_steps=100, tol=1e-5, sig=0.1, lbd=0.01, stepsize=0.001, rho=2, true_beta=None):
        """Gradient ascending"""
        self.update = 1
        while self.update < max_steps:
            print(f"######## [Update {self.update}] ########")
            self.steps = 0
            self.gradient = np.Inf
            while (norm(self.gradient) > tol) and (self.steps < max_steps):
                print(f"######## [Update {self.update}'s Step {self.steps}] ########")
                # gradient
                self.gradient = self.derivative_calcu(order=1) / self.n
                self.gradient = -self.gradient  # change into minimize problem
                # penalty for $\|B^\top B\| = 1$
                penalty = np.zeros_like(self.gradient)
                penalty[0:(self.p * self.K)] = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
                self.gradient += penalty
                # update theta
                self.theta = self.theta - stepsize * self.gradient
                self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
                self.beta, self.sigma = self.copy_beta_sigma()
                print(f"norm(gradient): {norm(self.gradient):.5f}")
                print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta):.7f}")
                self.steps += 1

            eq_violance = (norm(self.beta) ** 2 - 1) * 0.5
            if eq_violance <= tol:
                break
            # update lambda (Lagrangian Multiplier)
            lbd = lbd + sig * eq_violance
            sig = sig * rho
            self.update += 1

        self.beta, self.sigma = self.copy_beta_sigma()
        return self.beta.ravel(), self.sigma

    def NR_alg(self, max_steps=10, tol=1e-5, sig=0.001, lbd=0.01, rho=2, off_diag=False, true_beta=None, penalty=True):
        self.steps = 0
        self.gradient = np.Inf
        while (norm(self.gradient) > tol) and (self.steps < max_steps):
            print(f"######## [Step {self.steps}] ########")
            # gradient & Hessian
            self.gradient, self.Hessian = self.derivative_calcu(order=2, off_diag=off_diag)
            self.gradient = -self.gradient / self.n
            self.Hessian = -self.Hessian / self.n
            # penalty for $\|B^\top B\| = 1$
            pen_grad, pen_hess = 0, 0
            if penalty:
                pen_grad = np.zeros_like(self.gradient)
                pen_grad[0:(self.p * self.K)] = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
                pen_hess = np.zeros_like(self.Hessian)
                for j in range(self.K * self.p):
                    for k in range(self.K * self.p):
                        if j == k:
                            pen_hess[j, k] = lbd + sig * (norm(self.beta) ** 2 - 1 + 2 * (self.beta.ravel()[j]) ** 2)
                        else:
                            pen_hess[j, k] = 2 * sig * (self.beta.ravel()[j]) * (self.beta.ravel()[k])
            theta_diff = -np.linalg.inv(self.Hessian + pen_hess) @ (self.gradient + pen_grad)
            self.theta = self.theta + theta_diff
            # self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
            self.beta, self.sigma = self.copy_beta_sigma()
            print(f"norm(gradient): {norm(self.gradient):.7f}")
            print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta):.7f}")
            self.steps += 1
            eq_violance = abs((norm(self.beta) ** 2 - 1) * 0.5)
            print(f"eq_violance: {eq_violance:.7f}")
            lbd = lbd + sig * eq_violance
            sig = sig * rho

        self.beta, self.sigma = self.copy_beta_sigma()
        return self.beta.ravel(), self.sigma

    # def NR_alg(self, max_steps=10, tol=1e-5, sig=0.1, lbd=0.001, rho=2, off_diag=False, penalty=True, true_beta=None):
    #     self.update = 0
    #     while self.update < max_steps:
    #         print(f"######## [Update {self.update}] ########")
    #         self.steps = 0
    #         self.gradient = np.Inf
    #         while (norm(self.gradient) > tol) and (self.steps < max_steps):
    #             print(f"######## [Update {self.update}'s Step {self.steps}] ########")
    #             # gradient & Hessian
    #             self.gradient, self.Hessian = self.derivative_calcu(order=2, off_diag=off_diag)
    #             self.gradient = -self.gradient / self.n
    #             self.Hessian = -self.Hessian / self.n
    #             # penalty for $\|B^\top B\| = 1$
    #             pen_grad, pen_hess = 0, 0
    #             # if penalty:
    #             #     pen_grad = np.zeros_like(self.gradient)
    #             #     pen_grad[0:(self.p * self.K)] = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
    #             #     # self.gradient += pen_grad
    #             #     pen_hess = np.zeros_like(self.Hessian)
    #             #     for j in range(self.K * self.p):
    #             #         for k in range(self.K * self.p):
    #             #             if j == k:
    #             #                 pen_hess[j, k] = lbd + sig * (norm(self.beta) ** 2 - 1 + 2 * (self.beta.ravel()[j]) ** 2)
    #             #             else:
    #             #                 pen_hess[j, k] = 2 * sig * (self.beta.ravel()[j]) * (self.beta.ravel()[k])
    #             theta_diff = -np.linalg.inv(self.Hessian + pen_hess) @ (self.gradient + pen_grad)
    #             self.theta = self.theta + theta_diff
    #             self.beta, self.sigma = self.copy_beta_sigma()
    #             print(f"norm(gradient): {norm(self.gradient):.7f}")
    #             print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta):.7f}")
    #             self.steps += 1
    #         eq_violance = abs((norm(self.beta) ** 2 - 1) * 0.5)
    #         print(f"eq_violance: {eq_violance:.7f}")
    #         if eq_violance <= tol:
    #             break
    #         # update lambda (Lagrangian Multiplier)
    #         lbd = lbd + sig * eq_violance
    #         sig = sig * rho
    #         self.update += 1
    #
    #     self.beta, self.sigma = self.copy_beta_sigma()
    #     return self.beta.ravel(), self.sigma

    def compute_A_diff(self):
        self.p_ikm = self.compute_pikm()       # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)

    def derivative_calcu(self, order=1, off_diag=False):
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

        # gradient
        gradient = np.zeros_like(self.theta)
        gradient[:(self.K * self.p)] = partial_beta.ravel()
        gradient[(self.K * self.p):] = partial_sigma
        if order == 1:
            return gradient

        ##################################### 2st derivative #####################################
        A11 = np.zeros((self.K * self.p, self.K * self.p))  # partial beta^2: (pK, pK)
        A22 = -2 * partial_sigma / self.sigma               # partial sigma^2 (M, 1)
        A12 = np.zeros((self.K * self.p, self.M))           # partial beta partial sigma (pK, M)
        for j in range(self.K):
            for k in range(self.K):
                App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                   # (n, M)
                Sigma_jk = -App / (self.sigma.reshape((1, self.M)) ** 2)               # (n, M)
                Sigma_jk = Sigma_jk.reshape((self.n, self.M, 1, 1))                  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((self.n, 1, self.p, self.p))  # (n, M, p, p)
                # A11
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
                # A22 & A12
                Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                for m in range(self.M):
                    A22[m] += self.beta[j] @ Sigma_jk[m] @ self.beta[k] / self.sigma[m] ** 2

        # A12  (Kp, M)
        if off_diag:
            delta = -A_diff / self.sigma.reshape(1, 1, self.M) ** 2  # (n, K, M)
            for m in range(self.M):
                delta_m = np.transpose(delta[:, :, m])               # (K, n)
                A12[:, m] = (delta_m @ self.X).ravel()               # (K, p) -> (pK,)
            for j in range(self.K):
                for k in range(self.K):
                    App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                    App = self.A * App                                                    # (n, M)
                    Sigma_jk = App / (self.sigma.reshape((1, self.M)) ** 3)               # (n, M)
                    for m in range(self.M):
                        Sigma_jkm = Sigma_jk[:, m].reshape(self.n, 1, 1)                  # (n, 1, 1)
                        Sigma_jkm = Sigma_jkm * self.XXT                                  # (n, p, p)
                        Sigma_jkm = Sigma_jkm.sum(axis=0)                                 # (p, p)
                        Sigma_jkm = Sigma_jkm @ self.beta[k]                              # (p, 1)
                        A12[(j * self.p):((j + 1) * self.p), m] += Sigma_jkm
        matrix = np.zeros((self.K * self.p + self.M, self.K * self.p + self.M))
        matrix[0:(self.K * self.p), 0:(self.K * self.p)] = A11
        matrix[(self.K * self.p):(self.K * self.p + self.M), (self.K * self.p):(self.K * self.p + self.M)] = np.diag(A22)
        matrix[0:(self.K * self.p), (self.K * self.p):(self.K * self.p + self.M)] = A12
        matrix[(self.K * self.p):(self.K * self.p + self.M), 0:(self.K * self.p)] = np.transpose(A12)
        return gradient, matrix




