#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 23:27
# @Author  : Helenology
# @Site    : 
# @File    : generate_data.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.model_selection import train_test_split


def generate_data(K, p, N, n, M, alpha, seed=0):
    np.random.seed(seed=seed)

    # parameters - beta
    beta = np.random.randn(K+1, p)
    beta[0] = 0
    beta_norm = norm(beta)
    beta = beta / beta_norm

    # features - X
    X = np.random.randn(N, p) * 2  # (sub)-gaussian features
    X[:, 0] = 1                # set the first columns of X to be constants

    # true labels - Y
    Y = np.argmax(X.dot(np.transpose(beta)), axis=1)
    print("True Labels", pd.value_counts(Y), "\n")

    # pilot sample - X1 and Y1
    X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=(N-n)/N, random_state=seed)

    # annotator sigma
    sigma_list = np.arange(0.1, 4.1, 4/M)
#     sigma_list = np.ones(M) * 2 #np.arange(0.1, 5.1, 5/M)
#     sigma_list[0:int(M/2)] = 0.1

    # parameter vector
    theta = np.zeros(K * p + M)
    theta[:(p*K)] = beta[1:].ravel()
    theta[(p*K):] = sigma_list.reshape(-1)

    # annotation task assignment - A1
    A1 = np.random.binomial(1, alpha, size=(n, M))

    # annotation probability - AP1
    AP1 = X1.dot(np.transpose(beta))
    AP1 = AP1.reshape(AP1.shape + (1,))
    AP1 = AP1 / sigma_list
    AP1 = np.exp(AP1)
    AP1_sum = AP1.sum(axis=1, keepdims=True)
    AP1 = AP1 / AP1_sum

    # annotation - AY1
    AY1 = np.zeros((n, M))
    for i in range(n):
        for m in range(M):
            prob_im = AP1[i, :, m]
            Y_im = np.argmax(np.random.multinomial(1, prob_im, 1))
            AY1[i, m] = Y_im
    AY1[A1 == 0] = -1
    # print(AY1[0:2])

    return beta, sigma_list, theta, X, Y, X1, X2, Y1, Y2, A1, AY1
