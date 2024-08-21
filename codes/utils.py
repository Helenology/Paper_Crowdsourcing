#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 14:48
# @Author  : Helenology
# @Site    : 
# @File    : utils.py
# @Software: PyCharm


import numpy as np
import scipy


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


def compute_MaxMix_i(Xi, beta, Avar, n, M, alpha_n, K, p):
    """"""
    scale = np.sqrt(n * M * alpha_n)
    MaxMis_list = []
    for j in range(K+1):
        for k in range(j+1, K+1):
            Avar_jk = get_Avar_jk(Avar, p, j, j) + get_Avar_jk(Avar, p, k, k) \
                      - get_Avar_jk(Avar, p, j, k) - get_Avar_jk(Avar, p, k, j)
            MaxMis = Phi(- scale * np.abs(Xi @ (beta[j] - beta[k])) / np.sqrt(Xi @ Avar_jk @ Xi))  # Equation (2.11)
            MaxMis_list.append(MaxMis)
    MaxMis = max(MaxMis_list)
    return MaxMis


def compute_Mi(Xi, yi, beta, Avar, n, M, alpha_n, K, p, w):
    """"""
    ki_hat = yi
    scale = (scipy.stats.norm.ppf(w)) ** 2 / (n * M * alpha_n)
    Mi_list = []
    for k in range(K+1):
        if k == ki_hat:
            continue
        Avar_kik = get_Avar_jk(Avar, p, ki_hat, ki_hat) + get_Avar_jk(Avar, p, k, k) \
                  - get_Avar_jk(Avar, p, ki_hat, k) - get_Avar_jk(Avar, p, k, ki_hat)
        Mi = scale * (Xi @ Avar_kik @ Xi) / ((Xi @ (beta[ki_hat] - beta[k])) ** 2)
        Mi_list.append(Mi)
    Mi = max(Mi_list)
    return Mi