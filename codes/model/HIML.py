#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/26 21:01
# @Author  : Helenology
# @Site    : 
# @File    : HIML.py
# @Software: PyCharm

import numpy as np
from numpy.linalg import norm
from copy import copy
from model.BaseModel import BaseModel
from model.Initial import Initial
from model.MS import MS
from utils import *
import pandas as pd
from sklearn.linear_model import LogisticRegression


class HIML(MS):
    def __init__(self, X, Y, A, K, alpha=1):
        MS.__init__(self, X, Y, A, K, alpha, beta=None, sigma=None)
        # initial parameters
        self.beta, self.sigma, _ = self.init_param()

    def init_param(self):
        if self.K > 1:  # Multi-Logistic Regression with classes (K+1) > 2
            initial_b = np.zeros((self.M, self.K + 1, self.p))
            initial_beta = np.zeros((self.K + 1, self.p))
            initial_sigma = np.zeros(self.M)
            for m in range(self.M):
                y_m = self.Y[:, m]
                idx = (self.A[:, m] != 0)
                X_m = self.X[idx]
                y_m = y_m[idx]
                clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_m, y_m)
                initial_b[m] = clf.coef_         # Initial Estimator: $\widetilde B_m$ but with class 0
                initial_b[m] -= initial_b[m, 0]  # get rid of class 0's coefficient
                initial_sigma[m] = 1 / norm(initial_b[m])
                initial_beta += initial_b[m] * initial_sigma[m] / self.M
            initial_b = initial_b[:, 1:, :]  # (M, K, p)
            initial_beta = initial_beta[1:]
            initial_beta /= norm(initial_beta)  # normalize
            return initial_beta, initial_sigma, initial_b
        elif self.K == 1:  # 2Class-Logistic Regression with classes (K+1) = 2
            initial_b = np.zeros((self.M, self.K, self.p))
            initial_beta = np.zeros((self.K, self.p))
            initial_sigma = np.zeros(self.M)
            for m in range(self.M):
                y_m = self.Y[:, m]
                idx = (self.A[:, m] != 0)
                X_m = self.X[idx]
                y_m = y_m[idx]
                clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_m, y_m)
                initial_b[m] = clf.coef_                                  # Initial Estimator: $\widetilde B_m$
                initial_sigma[m] = 1 / norm(initial_b[m])                 # Initial Estimator: $\widetilde \sigma_m$
                initial_beta += initial_b[m] * initial_sigma[m] / self.M  # Initial Estimator: $\widetilde \beta$
            initial_beta /= norm(initial_beta)                            # re-scale to 1
            return initial_beta, initial_sigma, initial_b

    def fit(self):
        # TS Model
        beta, sigma = self.update_alg(max_steps=2)
        self.beta = beta
        self.sigma = sigma
        self.beta_hat = np.zeros((self.K + 1, self.p))
        self.beta_hat[1:] = beta.reshape(self.K, self.p)
        self.sigma_hat = copy(self.sigma)

        # Check Crowd Annotator Quality
        if (sigma < 0).sum() > 0:
            print(f"Crowd Annatator Error: there are spammers! Return their indexes!")
            return np.where(sigma < 0)[0]

    def predict_with_MaxMis(self, X):
        Y_hat = np.argmax(X.dot(np.transpose(self.beta_hat)), axis=1)
        alpha_n = np.mean(self.alpha)
        Avar = self.compute_Avar()
        MaxMis = [compute_MaxMix_i(X[i], self.beta_hat, Avar, self.n, self.M, alpha_n, self.K, self.p) for i in range(X.shape[0])]
        MaxMis = np.array(MaxMis)
        return Y_hat, MaxMis
