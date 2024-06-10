#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/21 16:54
# @Author  : Helenology
# @Site    : 
# @File    : Initial.py
# @Software: PyCharm

from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.linalg import norm


class Initial:
    def __init__(self, X, Y, A, K):
        self.steps = 0
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.M = Y.shape[1]
        self.K = K
        ######################## data preparation
        self.X = X
        self.Y = Y
        self.A = A
        self.initial_b = np.zeros((self.M, self.K, self.p))
        self.initial_beta = np.zeros((self.K, self.p))
        self.initial_sigma = np.zeros(self.M)

    def opt_alg(self):
        for m in range(self.M):
            y_m = self.Y[:, m]
            idx = (self.A[:, m] != 0)
            X_m = self.X[idx]
            y_m = y_m[idx]
            clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_m, y_m)
            self.initial_b[m] = clf.coef_
            self.initial_sigma[m] = 1 / norm(self.initial_b[m])
            self.initial_beta += self.initial_b[m] * self.initial_sigma[m] / self.M



