#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 21:17
# @Author  : Helenology
# @Site    : 
# @File    : INR.py
# @Software: PyCharm

import os
import sys
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.discrete.discrete_model import Probit
import copy
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../data/'))
sys.path.append(os.path.abspath('../model/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
from utils import *


class INR:
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        # beta initialization
        self.beta_initial = None
        self.beta_hat = None
        self.initialize_beta()
        # self.beta_initial = np.random.rand(self.p, 1)  # np.ones((p, 1))
        # self.beta_hat = copy.copy(self.beta_initial)
        # sigma initialization
        self.sigma_initial = np.ones(self.M)  # np.random.rand(self.M)
        self.sigma_initial[0] = 1
        self.sigma_hat = copy.copy(self.sigma_initial)
        print(f"================= INR Algorithm =================")

    def initialize_beta(self):
        A1 = self.A[:, 0]
        Y1 = self.Y[A1 == 1, 0]
        X1 = self.X[A1 == 1, :]
        model = Probit(Y1, X1)
        probit_model = model.fit(disp=0)
        # print(probit_model.summary())
        beta_hat = probit_model.params.reshape(-1, 1)
        self.beta_initial = beta_hat
        self.beta_hat = copy.copy(beta_hat)

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
        self.beta_hat = new_beta_hat
        return beta_mse

    def update_sigma(self):
        U = np.dot(self.X, self.beta_hat) / self.sigma_hat
        phi, Phi, Phi_minus = compute_phi_Phi(U)
        phi_dot = phi_dot_function(U)
        Delta = (self.Y - Phi) / Phi / Phi_minus
        rho, rho_minus = rho_function(phi, Phi, Phi_minus)

        # update sigma - first item
        tmp_1 = self.A * U * (self.Y * rho - (1 - self.Y) * rho_minus) / (self.sigma_hat ** 2)
        tmp_1 = 2 * tmp_1.sum(axis=0)

        # update sigma - second item
        tmp_2 = self.A * U ** 2 * (Delta * phi_dot - (Delta * phi) ** 2) / (self.sigma_hat ** 2)
        tmp_2 = tmp_2.sum(axis=0)

        # update sigma - third item
        tmp_3 = self.A * U * (- self.Y * rho + (1 - self.Y) * rho_minus) / self.sigma_hat
        tmp_3 = tmp_3.sum(axis=0)

        # update sigma
        new_sigma_hat = self.sigma_hat - 1 / (tmp_1 + tmp_2) * tmp_3
        new_sigma_hat[0] = 1
        sigma_mse = mean_squared_error(self.sigma_hat[1:], new_sigma_hat[1:])
        self.sigma_hat = new_sigma_hat
        return sigma_mse

    def INR_algorithm(self, maxIter=100, epsilon=1e-3, mseWarn=50):
        for i in range(maxIter):
            print(f'----------------- Iteration {i} -----------------')
            beta_mse = self.update_beta()
            print(f"update_beta: beta mse({beta_mse:.4f})")
            print(f"beta_hat: {self.beta_hat.reshape(-1)}")
            sigma_mse = self.update_sigma()
            print(f"update_sigma: sigma mse({sigma_mse:.4f})")
            print(f"sigma_hat: {self.sigma_hat}")
            mse = beta_mse + sigma_mse
            if mse < epsilon:  # the change of this and last step is small enough to stop the INR algorithm
                print(f"Success with mean square change: {mse:.4f}")
                return mse
            if np.any(self.sigma_hat < 0) or sigma_mse > mseWarn or \
                    (np.abs(self.sigma_hat - self.sigma_initial) < epsilon).sum() > 0:
                self.reinitialize_sigma(epsilon)

        print(f"Warning: reach the maxIter({maxIter}) with MSE: {mse:.4f}")
        return mse

    def reinitialize_sigma(self, epsilon=1e-3):
        print(f'++++++++++++ reinitialize_sigma +++++++++++++++')
        print("reinitialize_sigma:")
        # seldom changed index
        seldom_change_index = np.abs(self.sigma_hat - self.sigma_initial) < epsilon
        if np.any(seldom_change_index):
            self.sigma_initial[seldom_change_index] = self.sigma_initial[seldom_change_index] * 2
            self.sigma_hat[seldom_change_index] = self.sigma_initial[seldom_change_index]
            print(f"\tseldom changed index")

        # if some sigma_hat < 0, then the initialization of it should be smaller
        neg_index = self.sigma_hat < 0
        if np.any(neg_index):
            self.sigma_initial[neg_index] = self.sigma_initial[neg_index] / 2
            self.sigma_hat[neg_index] = self.sigma_initial[neg_index]
            print(f"\tnegative index")  # : {self.sigma_hat}

        # if sigma's change is too fast, then the initialization should be larger
        fast_change_index = self.sigma_hat > 10 * self.sigma_initial
        if np.any(fast_change_index):
            self.sigma_initial[fast_change_index] = self.sigma_initial[fast_change_index] / 2
            self.sigma_hat[fast_change_index] = self.sigma_initial[fast_change_index]
            print(f"\tfast change index")
        self.sigma_hat[0] = 1
        print(f"sigma_hat: {self.sigma_hat}")
        print(f'+++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == '__main__':
    N = 1000000
    p = 20
    M = 100
    seed = 0
    np.random.seed(seed)  # set random seed
    np.set_printoptions(precision=3)  # 设置小数位置为3位

    beta_star = np.ones(p)  # the true parameter of interest
    sigma_star = np.ones(M)
    # sigma_star[1:] *= np.arange(start=0.1, stop=10.1, step=(10 / M))[:(-1)]
    sigma_star[1:int(M / 2)] *= 0.1
    sigma_star[int(M / 2):] *= 10
    print(f"true sigma: {sigma_star}")
    X, Y_true = construct_synthetic_dataset(N, p, beta_star, seed=0)  # generate synthetic dataset
    alpha_list = [0.1] * M
    A_annotation, Y_annotation = synthetic_annotation(X, beta_star, M, sigma_star, alpha_list, seed=seed)

    t1 = time.time()
    inr = INR(X, Y_annotation, A_annotation)
    inr.INR_algorithm(maxIter=50)
    t2 = time.time()
    print("=================================")
    print(f"Time: {t2 - t1:.4f}")
    print("------------- beta --------------")
    print(f"true beta: {beta_star}")
    print(f"estimate beta: {inr.beta_hat.reshape(-1)}")
    print(f"final MSE: {mean_squared_error(beta_star, inr.beta_hat):.4f}")
    print("------------- sigma -------------")
    print(f"true sigma: {sigma_star}")
    print(f"estimate sigma: {inr.sigma_hat}")
    print(f"final MSE: {mean_squared_error(sigma_star, inr.sigma_hat):.4f}")
