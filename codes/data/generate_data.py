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
    """

    :param K: (K+1) is the number of classes
    :param p: dimension of features
    :param N: the number of instances
    :param n: the number of pilot samples
    :param M: the number of crowd annotators
    :param alpha: assignment probability; (1,) => equal probability; (M,) => individual-wise probability
    :param seed: random seed
    :return:
    """
    np.random.seed(seed=seed)

    # parameters - beta
    beta = np.random.randn(K+1, p)
    beta[0] = 0
    beta_norm = norm(beta)
    beta = beta / beta_norm

    # features - X
    X = np.random.randn(N, p) * 4  # (sub)-gaussian features
    X[:, 0] = 1                    # set the first columns of X to be constants

    # true labels - Y
    Y = np.argmax(X.dot(np.transpose(beta)), axis=1)
    print("True Labels", pd.value_counts(Y), "\n")

    # pilot sample - X1 and Y1
    ids = np.arange(N)
    X1, X2, Y1, Y2, pilot_ids, rest_ids = train_test_split(X, Y, ids, test_size=(N-n)/N, random_state=seed)

    # annotator sigma
    # sigma = np.arange(0.1, 4.1, 4/M)
    sigma = np.ones(M) * 2
    sigma[0:int(M/2)] = 0.5

    # parameter vector
    theta = np.zeros(K * p + M)
    theta[:(p*K)] = beta[1:].ravel()
    theta[(p*K):] = sigma.reshape(-1)

    # annotation task assignment - A1
    A1 = np.random.binomial(1, alpha, size=(n, M))

    # annotation probability - AP1
    AP1 = X1.dot(np.transpose(beta))          # (n, K+1)
    AP1 = AP1.reshape(AP1.shape + (1,))       # (n, K+1, 1)
    AP1 = AP1 / sigma                         # (n, K+1, M)
    AP1 = np.exp(AP1)                         # (n, K+1, M)
    AP1_sum = AP1.sum(axis=1, keepdims=True)  # (n, 1, M)
    AP1 = AP1 / AP1_sum                       # (n, K+1, M)

    # annotation - AY1
    AY1 = np.zeros((n, M))
    for i in range(n):
        for m in range(M):
            if A1[i, m] == 0:                 # The ith instance is [Not Assigned] to the mth crowd annotator.
                AY1[i, m] = -1
            else:                             # The ith instance is [Assigned] to the mth crowd annotator.
                prob_im = AP1[i, :, m]
                Y_im = np.argmax(np.random.multinomial(1, prob_im, 1))
                AY1[i, m] = Y_im
    AY1[A1 == 0] = -1

    return beta, sigma, theta, X, Y, X1, X2, Y1, Y2, A1, AY1, pilot_ids, rest_ids
