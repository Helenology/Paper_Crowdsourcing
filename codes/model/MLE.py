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
        self.theta = np.ones((self.K * self.p + self.M, )) * 1
        self.theta[:(self.K * self.p)] = np.random.randn(self.K * self.p)
        self.theta[:(self.K * self.p)] /= norm(self.theta[:(self.K * self.p)])
        self.beta, self.sigma = self.copy_beta_sigma()
        # optimization initialization
        self.gradient = None
        self.Hessian = None
        self.steps = 0

    def compute_likelihood(self):
        p_ikm = np.zeros((self.n, (self.K+1), self.M))  # (n, (K+1), M)
        p_ikm[:, 1:] = self.compute_pikm()
        p_ikm[:, 0] = 1 - p_ikm[:, 1:].sum(axis=1)      # p_ikm += 1e-5 # p_ikm /= p_ikm.sum(axis=1, keepdims=True)
        Y_onehot = np.ones((self.n, (self.K+1), self.M))
        for k in range(self.K):
            Y_onehot[:, k, :] = (self.Y == k).astype(int)
        likelihood = self.A.reshape(self.n, 1, self.M) * Y_onehot * np.log(p_ikm)
        likelihood = likelihood.sum() / self.n
        return likelihood

    def compute_XXT(self):
        """Conpute $X_i X_i^\top$ for $1 \leq i \leq n$"""
        XXT = (self.X.reshape(self.n, self.p, 1)) * (self.X.reshape(self.n, 1, self.p))
        return XXT

    def compute_Y_onehot(self):
        Y_onehot = np.ones((self.n, self.K, self.M))                      # here we neglect class 0
        Y_onehot *= self.Y.reshape(self.n, 1, self.M)
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

    def GD_alg(self, max_steps=10, epsilon=1e-5, lbd=0.1):
        while True:
            self.steps += 1
            likelihood = self.compute_likelihood()
            self.likelihood_list.append(likelihood)
            # gradient
            self.gradient = self.first_derivative_calcu()
            theta_diff = lbd * norm(self.gradient)
            self.theta = self.theta + lbd * self.gradient  # Maximum it because MLE
            theta_norm = norm(self.theta[:(self.K * self.p)])
            self.theta[:(self.K * self.p)] /= theta_norm
            self.theta[(self.K * self.p):] /= theta_norm
            self.beta, self.sigma = self.copy_beta_sigma()
            print('theta[0:10]', self.theta[0:10])
            print(f"[step {self.steps}] with likelihood: {likelihood}; theta diff: {theta_diff: .6f}")
            if (theta_diff < epsilon) or (self.steps > max_steps):
                break
        theta_norm = norm(self.theta[:(self.K * self.p)])
        self.theta[:(self.K * self.p)] /= theta_norm
        self.theta[(self.K * self.p):] /= theta_norm
        self.beta, self.sigma = self.copy_beta_sigma()

    def NR_alg(self, max_steps=10, epsilon=1e-5, lbd=0.1):
        while True:
            self.steps += 1
            # gradient and penalty
            self.gradient = self.first_derivative_calcu()
            self.Hessian = self.second_derivative_calcu()
            penalty = np.zeros_like(self.theta)  # penalty
            penalty[:(self.K * self.p)] = self.beta.ravel() * lbd

            print(f"theta_t:", self.theta)
            print(f"gradient:", self.gradient)
            theta_new = self.theta - np.linalg.inv(self.Hessian).dot(self.gradient)   # Maximum it because MLE
            theta_diff = norm(self.theta - theta_new)
            self.theta = theta_new
            self.beta, self.sigma = self.copy_beta_sigma()
            print(f"[step {self.steps}] with theta diff: {theta_diff: .6f}")
            if (theta_diff < epsilon) or (self.steps > max_steps):
                break
        theta_norm = norm(self.theta[:(self.K * self.p)])
        self.theta[:(self.K * self.p)] /= theta_norm
        self.theta[(self.K * self.p):] /= theta_norm
        self.beta, self.sigma = self.copy_beta_sigma()

    def first_derivative_calcu(self):
        # partial beta
        p_ikm = self.compute_pikm()
        self.p_ikm = p_ikm
        diff = self.Y_onehot - p_ikm
        self.diff = diff
        A = self.A.reshape(self.n, 1, self.M)
        delta = diff * A / self.sigma
        delta = delta.sum(axis=2)
        partial_beta = np.transpose(delta).dot(self.X)
        # partial sigma
        delta = diff * A
        remain = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        remain = remain / (self.sigma**2)
        partial_sigma = -delta * remain
        partial_sigma = partial_sigma.sum(axis=(0, 1))
        # gradient
        gradient = np.zeros_like(self.theta)
        gradient[:(self.K * self.p)] = partial_beta.ravel()
        gradient[(self.K * self.p):] = partial_sigma
        gradient /= self.n
        return gradient

    def second_derivative_calcu(self):
        # partial beta partial beta^T
        A11 = np.zeros((self.K * self.p, self.K * self.p))
        for j in range(self.K):
            for k in range(self.K):
                delta1 = 0
                if j == k:
                    delta1 = self.A * self.p_ikm[:, j, :]
                delta2 = -self.A * self.p_ikm[:, j, :] * self.p_ikm[:, k, :]
                delta = -(delta1 + delta2)
                delta = delta / (self.sigma**2)
                delta = delta.sum(axis=1).reshape(self.n, 1, 1)
                delta = delta * self.XXT
                delta = delta.sum(axis=0)
                A11[(j * self.p):((j+1) * self.p), (k * self.p):((k+1) * self.p)] = delta

        # partial sigma partial sigma^T
        diff = self.diff
        A22 = np.zeros((self.M, self.M))
        delta = diff * self.A.reshape(self.n, 1, self.M)
        remain = 2 * self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1)
        remain = remain / (self.sigma ** 3)
        partial_sigma = (delta * remain).sum(axis=(0, 1))
        A22 += np.diag(partial_sigma)

        a = self.X.dot(np.transpose(self.beta)).reshape(self.n, self.K, 1) / (self.sigma**2)
        a = a**2
        b = (self.A.reshape(self.n, 1, self.M) * self.p_ikm * a).sum(axis=(0, 1))
        A22 -= np.diag(b)

        tmp = np.zeros((self.n, self.M))
        for j in range(self.K):
            for k in range(self.K):
                tmp_value = self.A * self.p_ikm[:, j, :] * self.p_ikm[:, k, :]
                a = (self.X.dot(self.beta[j].reshape(self.p, 1)) / self.sigma**2)
                b = (self.X.dot(self.beta[k].reshape(self.p, 1)) / self.sigma**2)
                tmp += tmp_value * a * b
        tmp = tmp.sum(axis=0)
        A22 += np.diag(tmp)

        matrix = np.zeros((self.K * self.p + self.M, self.K * self.p + self.M))
        matrix[:(self.K * self.p), :(self.K * self.p)] = A11
        matrix[(self.K * self.p):, (self.K * self.p):] = A22
        matrix /= self.n
        return matrix


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
    # mle_model.first_derivative_calcu()
    # mle_model.second_derivative_calcu()
    likelihood = mle_model.compute_likelihood()





