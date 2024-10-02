#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/11 15:44
# @Author  : Helenology
# @Site    : 
# @File    : MS.py
# @Software: PyCharm

from model.BaseModel import BaseModel
import numpy as np
from numpy.linalg import norm
from copy import copy
from utils import *


class MS(BaseModel):
    def __init__(self, X, Y, A, K, alpha, beta, sigma):
        BaseModel.__init__(self, X, Y, A, K, alpha)
        # parameter initialization
        self.init_beta = copy(beta)
        self.init_sigma = copy(sigma)
        self.beta = copy(beta)
        self.sigma = copy(sigma)
        # optimization initialization
        self.steps = 0
        # Avar (pK, pK)
        self.A11 = None
        self.A11_inv = None
        self.Avar = None

    def compute_derivative(self, tmp_beta, tmp_sigma, fixsigma=False):
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
        if fixsigma is False:  # update theta = (beta, sigma)
            ##################################### 1st derivative #####################################
            # partial beta
            p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)  # (n, K, M)
            A_diff = self.compute_A_diff(p_ikm)
            delta = A_diff / tmp_sigma.reshape(1, 1, M)  # (n, K, M)
            delta = delta.sum(axis=2)  # (n, K)
            partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)
            # partial sigma
            partial_sigma = - A_diff / tmp_sigma.reshape(1, 1, M) ** 2  # (n, K, M)
            partial_sigma *= (self.X @ np.transpose(tmp_beta)).reshape((n, K, 1))  # (n, K, M)
            partial_sigma = partial_sigma.sum(axis=(0, 1))  # (M,)
            ##################################### 2st derivative #####################################
            A11 = np.zeros((K * p, K * p))  # partial beta^2:  (pK, pK)
            A22 = - 2 * partial_sigma / tmp_sigma  # partial sigma^2: (M, 1)
            for j in range(K):
                for k in range(K):
                    App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]  # (n, M)
                    App = self.A * App  # (n, M)
                    Sigma_jk = - App / (tmp_sigma.reshape(1, M) ** 2)  # (n, M)
                    Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))  # (n, M, 1, 1)
                    Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))  # (n, M, p, p)
                    A11[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
                    Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                    for m in range(M):
                        A22[m] += tmp_beta[j] @ Sigma_jk[m] @ tmp_beta[k] / tmp_sigma[m] ** 2
            return partial_beta.ravel(), A11, partial_sigma, A22
        else:  # Fix sigma, only update beta
            # partial beta
            p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)  # (n, K, M)
            A_diff = self.compute_A_diff(p_ikm)
            delta = A_diff / tmp_sigma.reshape(1, 1, M)  # (n, K, M)
            delta = delta.sum(axis=2)  # (n, K)
            partial_beta = np.transpose(delta) @ self.X  # (K, n) @ (n, p) = (K, p)
            # partial beta^2:  (pK, pK)
            A11 = np.zeros((K * p, K * p))
            for j in range(K):
                for k in range(K):
                    App = int(j == k) * (p_ikm[:, j, :]) - p_ikm[:, j, :] * p_ikm[:, k, :]  # (n, M)
                    App = self.A * App  # (n, M)
                    Sigma_jk = -App / (tmp_sigma.reshape(1, M) ** 2)  # (n, M)
                    Sigma_jk = Sigma_jk.reshape((n, M, 1, 1))  # (n, M, 1, 1)
                    Sigma_jk = Sigma_jk * self.XXT.reshape((n, 1, p, p))  # (n, M, p, p)
                    A11[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
            return partial_beta.ravel(), A11, None, None

    def update_alg(self, max_steps=1, true_beta=None, echo=True):
        while True:
            self.steps += 1
            # gradient & Hessian
            if self.steps == 1:
                partial_beta, A11, partial_sigma, A22 = self.compute_derivative(self.beta, self.sigma, fixsigma=False)
            else:
                partial_beta, A11, partial_sigma, A22 = self.compute_derivative(self.beta, self.sigma, fixsigma=True)
            # update beta
            partial_beta /= - self.n
            A11 /= - self.n
            try:
                A11_inv = np.linalg.inv(A11)
            except np.linalg.LinAlgError:
                print("The matrix A11 is singular. Using pseudo-inverse.")
                A11_inv = np.linalg.pinv(A11)
            beta_diff = - A11_inv @ partial_beta
            self.beta += beta_diff.reshape(self.K, self.p)
            self.beta /= norm(self.beta)
            # update sigma
            if partial_sigma is not None:
                partial_sigma /= - self.n
                A22 /= - self.n
                sigma_diff = - A22 ** (-1) * partial_sigma
                tmp_sigma = np.abs(self.sigma + sigma_diff)
                # tmp_sigma[tmp_sigma > 10] = 10
                # tmp_sigma[tmp_sigma < 1e-3] = 1e-3
                self.sigma = tmp_sigma
                # self.sigma += sigma_diff
                # self.sigma = np.abs(self.sigma)
            if echo:
                print(f"######## [Step {self.steps}] ########")
                beta_gradient = np.sqrt(norm(partial_beta) ** 2)
                print(f"norm(beta_gradient): {beta_gradient:.7f}")
                if partial_sigma is not None:
                    sigma_gradient = np.sqrt(norm(partial_sigma) ** 2)
                    print(f"norm(sigma_gradient): {sigma_gradient:.7f}")
                if true_beta is not None:
                    print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
            # terminal condition
            if self.steps >= max_steps:
                break

        self.partial_beta = partial_beta
        self.A11 = A11
        self.A11_inv = A11_inv
        self.sigma_diff = sigma_diff
        self.A22 = A22
        self.partial_sigma = partial_sigma
        return self.beta.ravel(), self.sigma

    def compute_Avar(self, tmp_beta, tmp_sigma):
        K = self.K
        M = self.M
        n = self.n
        p = self.p
        alpha_list = np.array(self.alpha) * 1.0
        alpha_n = np.mean(alpha_list)
        cm_list = alpha_list / alpha_n

        # Sigma & Sigma_m
        p_ikm = self.compute_pikm(tmp_beta, tmp_sigma)  # (n, K, M)
        Sigma = np.zeros((K * p, K * p))
        Sigma_m_list = np.zeros((M, K * p, K * p))
        for m in range(M):
            Sigma_m = np.zeros((K * p, K * p))  # $\Sigma_m^*$ with size of (pK, pK)
            for j in range(K):
                for k in range(K):
                    App = int(j == k) * (p_ikm[:, j, m]) - p_ikm[:, j, m] * p_ikm[:, k, m]  # (n, )
                    App = self.A[:, m] * App  # (n, )
                    Sigma_jk = App.reshape((n, 1, 1)) * self.XXT  # (n, p, p)
                    Sigma_m[(j * p):((j + 1) * p), (k * p):((k + 1) * p)] = Sigma_jk.sum(axis=0)  # (p, p)
            Sigma_m /= n * alpha_list[m]
            Sigma += cm_list[m] / (tmp_sigma[m] ** 2) * Sigma_m
            Sigma_m_list[m] = Sigma_m
        Sigma /= M

        # Sigma Inverse & D
        try:
            Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            print("The matrix Sigma is singular. Using pseudo-inverse.")
            Sigma_inv = np.linalg.pinv(Sigma)
        D = self.compute_D(tmp_beta)

        # $\Sigma^\text{TS}$
        Sigma_TS = Sigma * 1.0
        beta_vec = tmp_beta.reshape((K * p, 1))
        for m in range(M):
            Sigma_m = Sigma_m_list[m]
            tmp_val = cm_list[m] / (tmp_sigma[m] ** 2)
            tmp_val *= Sigma_m @ beta_vec @ np.transpose(beta_vec) @ Sigma_m
            tmp_val /= np.transpose(beta_vec) @ Sigma_m @ beta_vec
            Sigma_TS -= tmp_val / M
        # compute Avar
        Avar = D @ Sigma_inv @ Sigma_TS @ Sigma_inv @ D
        return Avar

    def get_Avar_jk(self, j, k):
        if j == 0 or k == 0:
            return 0
        p = self.p
        return self.Avar[((j - 1) * p):(j * p), ((k - 1) * p):(k * p)]
