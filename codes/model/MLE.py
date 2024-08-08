#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/20 15:14
# @Author  : Helenology
# @Site    : 
# @File    : MLE.py
# @Software: PyCharm

from model.BaseModel import BaseModel
import numpy as np
from numpy.linalg import norm

class MLE(BaseModel):
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

    def NR_alg(self, max_updates=1, max_steps=1, tol=1e-5, sig=0.01, lbd=0.01, rho=2, true_beta=None):
        np.random.seed(0)
        self.update = 0
        min_grad = np.Inf
        min_beta = None
        min_sigma = None
        while self.update < max_updates:
            print(f"######## [Update {self.update}] ########")
            self.steps = 0
            self.gradient = np.Inf
            beta_diff = np.Inf
            while (norm(self.gradient) > tol) and (self.steps < max_steps) and (norm(beta_diff) > tol):
                print(f"######## [Step {self.steps}] ########")
                # gradient & Hessian
                self.gradient, self.Hessian = self.compute_derivative(self.beta, self.sigma)
                self.gradient = -self.gradient / self.n
                self.Hessian = -self.Hessian / self.n
                # penalty for $\|B^\top B\| = 1$
                pen_grad = np.zeros_like(self.gradient)
                pen_grad[0:(self.p * self.K)] = (lbd + sig * (norm(self.beta) ** 2 - 1)) * self.beta.ravel()
                pen_hess = np.zeros_like(self.Hessian)
                for j in range(self.K * self.p):
                    for k in range(self.K * self.p):
                        if j == k:
                            pen_hess[j, k] = lbd + sig * (norm(self.beta) ** 2 - 1 + 2 * (self.beta.ravel()[j]) ** 2)
                        else:
                            pen_hess[j, k] = 2 * sig * (self.beta.ravel()[j]) * (self.beta.ravel()[k])
                # update theta
                theta_diff = -np.linalg.inv(self.Hessian + pen_hess) @ (self.gradient + pen_grad)
                self.theta = self.theta + theta_diff
                # self.theta[0:(self.K * self.p)] /= norm(self.theta[0:(self.K * self.p)])
                self.beta, self.sigma = self.copy_beta_sigma()
                curr_grad = norm(self.gradient)
                if curr_grad < min_grad:
                    print(f"[Record parameter] grad:{min_grad:.6f}->{curr_grad:.6f}")
                    min_grad = norm(self.gradient)
                    min_beta = self.beta
                    min_sigma = self.sigma

                print(f"norm(gradient): {curr_grad:.7f}")
                if true_beta is not None:
                    print(f"RMSE(beta): {norm(self.beta.ravel() - true_beta.ravel()):.7f}")
                self.steps += 1
            eq_violance = abs((norm(self.beta) ** 2 - 1) * 0.5)
            if eq_violance <= tol:
                break
            # update lambda (Lagrangian Multiplier)
            lbd = lbd + sig * eq_violance
            sig = sig * rho
            self.update += 1

        self.beta, self.sigma = self.copy_beta_sigma()
        return min_beta.ravel(), min_sigma