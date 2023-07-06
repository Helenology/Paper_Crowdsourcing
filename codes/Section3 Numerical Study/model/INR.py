#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 21:17
# @Author  : Helenology
# @Site    : 
# @File    : INR.py
# @Software: PyCharm

import os
import sys
sys.path.append(os.path.abspath('../'))
import numpy as np
from utils import *
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath('../data/'))
sys.path.append(os.path.abspath('../model/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
from crowdsourcing_model import *


class INR:
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.beta_hat = np.random.rand(self.p, 1)  # np.ones((p, 1))
        self.sigma_hat = np.ones(self.M)  # np.random.rand(self.M)
        self.sigma_hat[0] = 1


    def update_beta(self):
        U = np.dot(self.X, self.beta_hat) / self.sigma_hat
        phi, Phi, Phi_minus = compute_phi_Phi(U)
        phi_dot = phi_dot_function(U)
        Delta = (self.Y - Phi) / Phi / Phi_minus
        rho, rho_minus = rho_function(phi, Phi, Phi_minus)

        # update beta - hessian inverse
        tmp_hess = self.A * (Delta * phi_dot - (Delta * phi)**2) / (self.sigma_hat**2)
        tmp_hess = tmp_hess.sum(axis=1).reshape(self.n, 1)
        tmp_hess = self.X.transpose().dot(self.X * tmp_hess)
        tmp_hess_inv = np.linalg.inv(tmp_hess)

        # update beta - score
        tmp_score = self.A * (self.Y * rho - (1 - self.Y) * rho_minus) / self.sigma_hat
        tmp_score = tmp_score.sum(axis=1).reshape(self.n, 1)
        tmp_score = self.X * tmp_score
        tmp_score = tmp_score.sum(axis=0)

        # update beta
        new_beta_hat = self.beta_hat - tmp_hess_inv.dot(tmp_score).reshape(-1, 1)
        beta_mse = mean_squared_error(new_beta_hat, self.beta_hat)
        print(f"beta mse: {beta_mse}")
        self.beta_hat = new_beta_hat
        return beta_mse

    def update_sigma(self):
        return 0

    # def update_sigma(self):
    #     print(f"initial sigma_hat: {self.sigma_hat}")
    #     while True:
    #         U = np.dot(self.X, self.beta_hat) / self.sigma_hat
    #         phi, Phi, Phi_minus = compute_phi_Phi(U)
    #         phi_dot = phi_dot_function(U)
    #         Delta = (self.Y - Phi) / Phi / Phi_minus
    #         rho, rho_minus = rho_function(phi, Phi, Phi_minus)
    #
    #         # update sigma - first item
    #         tmp_1 = self.A * U * (self.Y * rho - (1 - self.Y) * rho_minus) / (self.sigma_hat ** 2)
    #         tmp_1 = 2 * tmp_1.sum(axis=0)
    #
    #         # update sigma - second item
    #         tmp_2 = self.A * U**2 * (Delta * phi_dot - (Delta * phi) ** 2) / (self.sigma_hat ** 2)
    #         tmp_2 = tmp_2.sum(axis=0)
    #
    #         # update sigma - third item
    #         tmp_3 = self.A * U * (- self.Y * rho + (1 - self.Y) * rho_minus) / self.sigma_hat
    #         tmp_3 = tmp_3.sum(axis=0)
    #
    #         # update sigma
    #         new_sigma_hat = self.sigma_hat - 1 / (tmp_1 + tmp_2) * tmp_3
    #         new_sigma_hat[0] = 1
    #         mse = mean_squared_error(self.sigma_hat[1:], new_sigma_hat[1:])
    #         print(f"mse: {mse:.6f}")
    #         self.sigma_hat = new_sigma_hat
    #         if mse < 0.001:
    #             break

    def INR_algorithm(self, maxIter=100, epsilon=1e-3):
        for i in range(maxIter):
            beta_mse = self.update_beta()
            sigma_mse = self.update_sigma()
            mse = beta_mse + sigma_mse
            if mse < epsilon:
                print(f"Success with MSE: {mse:.6f}")
                return mse
        print(f"Reach the maxIter({maxIter}) with MSE: {mse:.6f}")
        return mse





if __name__ == '__main__':
    N = 100000
    p = 10
    M = 5
    seed = 0
    np.random.seed(seed)  # set random seed
    beta_star = np.ones(p)  # the true parameter of interest
    sigma_star = np.ones(M)
    X, Y_true = construct_synthetic_dataset(N, p, beta_star, seed=0)  # generate synthetic dataset
    alpha_list = [0.5] * M
    A_annotation, Y_annotation = synthetic_annotation(X, beta_star, M, sigma_star, alpha_list, seed=seed)

    inr = INR(X, Y_annotation, A_annotation)
    inr.INR_algorithm()
    print(inr.beta_hat)
    print(mean_squared_error(beta_star, inr.beta_hat))



    # inr.update_sigma()
    # print(inr.sigma_hat)
    # print(mean_squared_error(sigma_star, inr.sigma_hat))


