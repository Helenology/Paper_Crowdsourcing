#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 15:17
# @Author  : Helenology
# @Site    : 
# @File    : OS_Update.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import norm


class OS:
    def __init__(self, X, Y, A, K, beta=None, sigma=None):
        """

        :param X:
        :param Y:
        :param A:
        :param K:
        :param beta:
        :param sigma: annotator's parameter sigma
        """
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        self.beta = beta
        self.sigma = sigma
        # data preparation
        self.X = X
        self.XXT = self.compute_XXT()            # (n, p, p)
        self.Y = Y
        self.Y_onehot = self.compute_Y_onehot()  # (n, K, M)
        self.A = A

    def compute_XXT(self):
        """Conpute $X_i X_i^\top$ for $1 \leq i \leq n$"""
        XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))
        return XXT

    def compute_pikm(self):
        value_ik = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        value_ikm = value_ik / self.sigma.reshape(1, 1, self.M)
        value_ikm = np.exp(value_ikm)
        value_sum = value_ikm.sum(axis=1, keepdims=True) + 1  # +1 due to class 0
        p_ikm = value_ikm / value_sum
        return p_ikm

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
        for k in range(self.K):
            Y_onehot[:, k, :] = (Y_onehot[:, k, :] == (k + 1)).astype(int)  # here we neglect class 0
        return Y_onehot


    def compute_A_diff(self):
        self.p_ikm = self.compute_pikm()       # (n, K, M)
        diff = self.Y_onehot - self.p_ikm      # (n, K, M)
        A = self.A.reshape(self.n, 1, self.M)  # (n, 1, M)
        A_diff = A * diff                      # (n, K, M)
        return A_diff                          # (n, K, M)


    def derivative_calcu(self):
        ##################################### 1st derivative #####################################
        # partial beta
        print(f"[1st derivative] beta")
        A_diff = self.compute_A_diff()
        delta = A_diff / self.sigma.reshape(1, 1, self.M)  # (n, K, M)
        delta = delta.sum(axis=2)                          # (n, K)
        partial_beta = np.transpose(delta) @ self.X        # (K, n) @ (n, p) = (K, p)

        # partial sigma
        print(f"[1st derivative] sigma")
        partial_sigma = -A_diff / self.sigma.reshape(1, 1, self.M) ** 2  # (n, K, M)
        partial_sigma *= (self.X @ np.transpose(self.beta)).reshape((self.n, self.K, 1))  # (n, K, 1)
        partial_sigma = partial_sigma.sum(axis=(0, 1))     # (M,)

        ##################################### 2st derivative #####################################
        A11 = np.zeros((self.K * self.p, self.K * self.p))  # partial beta^2
        A22 = -2 * partial_sigma / self.sigma                        # partial sigma^2
        for j in range(self.K):
            for k in range(self.K):
                print(f"[2st derivative] ({j}, {k})")
                App = int(j == k) * (self.p_ikm[:, j, :]) - self.p_ikm[:, j, :] * self.p_ikm[:, k, :]  # (n, M)
                App = self.A * App                                                   # (n, M)
                Sigma_jk = -App / (self.sigma.reshape((1, self.M))**2)               # (n, M)
                Sigma_jk = Sigma_jk.reshape((self.n, self.M, 1, 1))                  # (n, M, 1, 1)
                Sigma_jk = Sigma_jk * self.XXT.reshape((self.n, 1, self.p, self.p))  # (n, M, p, p)
                # A11
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = Sigma_jk.sum(axis=(0, 1))  # (p, p)
                # A22
                Sigma_jk = Sigma_jk.sum(axis=0)  # (M, p, p)
                for m in range(self.M):
                    A22[m] += self.beta[j] @ Sigma_jk[m] @ self.beta[k] / self.sigma[m]**2

        return partial_beta, partial_sigma, A11, A22

    def one_step_update(self):
        gradient_beta, gradient_sigma, Hessian_beta, Hessian_sigma = self.derivative_calcu()
        # beta update
        gradient_beta = gradient_beta.ravel()
        beta = self.beta.ravel() - np.linalg.inv(Hessian_beta).dot(gradient_beta)
        # sigma update
        sigma = self.sigma - Hessian_sigma**(-1) * gradient_sigma
        return beta, sigma


if __name__ == "__main__":
    n = 100
    p = 10
    M = 5
    K = 1
    X = np.ones((n, p))
    Y = np.ones((n, M)) * 2
    Y[0] = -1
    A = np.ones((n, M))
    A[0] = 0

    beta = np.ones((K, p))
    sigma = np.ones(M)
    os = OS(X, Y, A, K, beta, sigma)
    beta = os.one_step_update()
    print(beta.shape)