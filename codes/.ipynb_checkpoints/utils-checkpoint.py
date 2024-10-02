#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 14:48
# @Author  : Helenology
# @Site    : 
# @File    : utils.py
# @Software: PyCharm


import numpy as np
import scipy
import pandas as pd


def compute_rmse(est, true):
    est = est.ravel()
    true = true.ravel()
    length = len(est)
    rmse = np.linalg.norm(est - true) / np.sqrt(length)
    return rmse


def Phi(x):
    return scipy.stats.norm.cdf(x)


def get_Avar_jk(Avar, p, j, k):
    if j == 0 or k == 0:
        return 0
    return Avar[((j - 1) * p):(j * p), ((k - 1) * p):(k * p)]


def compute_MaxMis_i(Xi, beta, Avar, n, M, alpha_n, K, p):
    """"""
    scale = np.sqrt(n * M * alpha_n)
    MaxMis_list = []
    for j in range(K+1):
        for k in range(j+1, K+1):
            Avar_jk = get_Avar_jk(Avar, p, j, j) + get_Avar_jk(Avar, p, k, k) \
                      - get_Avar_jk(Avar, p, j, k) - get_Avar_jk(Avar, p, k, j)
            MaxMis = - scale * np.abs(Xi @ (beta[j] - beta[k])) / np.sqrt(np.abs(Xi @ Avar_jk @ Xi))
            MaxMis_list.append(MaxMis)
    try:
        MaxMis = max(MaxMis_list)
    except:
        print(MaxMis_list)
    return MaxMis


