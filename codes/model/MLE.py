#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/19 18:12
# @Author  : Helenology
# @Site    : 
# @File    : MLE.py
# @Software: PyCharm


import numpy as np
from numpy.linalg import norm
import copy


class MLE:
    def __init__(self, X, Y, A, K):
        self.steps = 0
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        ######################## data preparation
        self.X = X
        self.Y = Y
        self.Y_onehot = self.compute_Y_onehot()  # (n, K, M)
        self.A = A

        ######################## parameter initialization
        self.theta = np.ones((self.K * self.p + self.M, )) * 0.1
        self.theta[(self.K * self.p):] = np.ones(self.M)
        self.beta, self.sigma = self.copy_beta_sigma()

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))                      # here we neglect class 0
        Y_onehot *= Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k+1)).astype(int)  # here we neglect class 0
        return Y_onehot

    def copy_beta_sigma(self):
        beta = self.theta[0:(self.p * self.K)].reshape(self.K, self.p)
        sigma = self.theta[(self.p * self.K):]
        return beta, sigma

    def compute_pikm(self):
        value_ik = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # Here +1 is because class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def Newton_Raphson_alg(self, max_steps=50, epsilon=1e-6):
        while True:
            self.steps += 1
            first_derivative = self.first_derivative_calcu()
            second_derivative = self.second_derivative_calcu()
            theta_new = self.theta - 1 / second_derivative * first_derivative
            theta_diff = norm(self.theta, theta_new)
            if (theta_diff < epsilon) or (self.steps > max_steps):
                break

    def first_derivative_calcu(self):
        # partial beta
        p_ikm = self.compute_pikm()
        diff = self.Y_onehot - p_ikm
        A = self.A.reshape(self.n, 1, self.M)
        delta = diff * A / self.sigma
        delta = delta.sum(axis=2)
        partial_beta = np.transpose(delta).dot(X)
        # partial sigma
        delta = diff * A
        remain = X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        remain = remain / (self.sigma**2)
        partial_sigma = -delta * remain
        partial_sigma = partial_sigma.sum(axis=(0, 1))
        # gradient
        gradient = np.zeros_like(self.theta)
        gradient[:(self.K * self.p)] = partial_beta.ravel()
        gradient[(self.K * self.p):] = partial_sigma
        return gradient

    def second_derivative_calcu(self):
        # partial beta partial beta^T
        pass


if __name__ == "__main__":
    n = 100
    p = 10
    M = 5
    K = 2
    X = np.ones((n, p))
    Y = np.ones((n, M)) * 2
    Y[0] = -1
    A = np.ones((n, M))
    A[0] = 0

    mle_model = MLE(X, Y, A, K)
    mle_model.first_derivative_calcu()




