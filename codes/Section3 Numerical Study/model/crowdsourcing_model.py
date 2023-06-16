#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/12 17:57
# @Author  : Helenology
# @Site    : 
# @File    : crowdsourcing_model.py
# @Software: PyCharm


import numpy as np
import sys
import os
from scipy.stats import norm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize, newton
from scipy.stats import norm
sys.path.append(os.path.abspath('../data/'))
from synthetic_annotators import *


def neg_loglikelihood(theta, X, Y_annotation, A_annotation):
    """
    Compute the negative log-likelihood.
    :param X: features (n x p)
    :param Y_annotation:
    :param A_annotation:
    :param theta: parameters of size (p+M-1)
    :return: negative log-likelihood
    """
    n, p = X.shape
    M = Y_annotation.shape[1]

    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list)
    assert len(all_sigma_list) == M
    all_sigma_list = all_sigma_list.reshape((1, -1))  # shape=(1, M)

    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list
    Phi_matrix = norm.cdf(inner_matrix)
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20  # otherwise it could lead to inf
    Phi_matrix1 = 1 - Phi_matrix
    Phi_matrix1[Phi_matrix1 < 1e-20] = 1e-20  # otherwise it could lead to inf
    log_likelihood = Y_annotation * np.log(Phi_matrix) + (1 - Y_annotation) * np.log(Phi_matrix1)
    log_likelihood = np.sum(log_likelihood[A_annotation == 1])
    log_likelihood = log_likelihood / Y_annotation.shape[0] / Y_annotation.shape[1]
    return -log_likelihood


def rho_function(inner_matrix):
    tmp = norm.pdf(inner_matrix) / norm.cdf(inner_matrix)
    return tmp


def score_function(X, Y_annotation, A_annotation, theta):  # , inner_matrix
    """

    :param X:
    :param Y_annotation:
    :param A_annotation:
    :param inner_matrix: (n * M) X.dot(beta) / all_sigma_list
    :return:
    """
    n, p = X.shape
    M = Y_annotation.shape[1]
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list)
    all_sigma_list = all_sigma_list.reshape((1, -1))  # shape=(1, M)

    X_sigma = X / all_sigma_list
    # score_vec = np.zeros(p+M-1)
    # # first-order derivative w.r.t. beta
    # rho_matrix = rho_function(inner_matrix)
    # rho_matrix_minus = rho_function(-inner_matrix)
    # common_value = Y_annotation * rho_matrix - (1 - Y_annotation) * rho_matrix_minus
    # common_matrix = common_value * A_annotation
    # score_beta = common_matrix.sum(axis=1).reshape((-1, 1))
    # score_beta = X.transpose().dot(score_beta)
    # score_vec[0:p] = score_beta.reshape(-1)
    # # first-order derivative w.r.t. sigma
    # score_sigma = -common_matrix
    # tmp = inner_matrix / all_sigma_list

    return X_sigma



    # Phi_matrix[Phi_matrix < 1e-20] = 1e-20  # otherwise it could lead to inf
    log_likelihood = Y_annotation * np.log(Phi_matrix) + (1 - Y_annotation) * np.log(Phi_matrix1)



def crowdsourcing_model(X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]
    theta_start = np.random.rand(p + M - 1)
    # use built-in module
    # res = minimize(neg_loglikelihood, theta_start, method='BFGS',
    #                args=(X, Y_annotation, A_annotation),
    #                options={'disp': True}
    #                )
    # Newton-Raphson optimization
    res = newton(neg_loglikelihood, theta_start,  # fprime=fder,
                 args=(X, Y_annotation, A_annotation), maxiter=100)
    return res


if __name__ == '__main__':
    N = 1000
    p = 10
    M = 10
    alpha_list = [1] * M
    X = np.random.randn(N, p)  # features
    X[:, 0] = 1  # set the first columns of X to be constants
    beta0 = np.random.rand(p)
    Y_true = (X.dot(beta0) > 0).astype(int)  # true labels
    sigma0_list = np.ones(M)
    seed = 0
    A1_annotation, Y1_annotation = synthetic_annotation(X, beta0, M, sigma0_list, alpha_list, 0, seed=seed)

    # maximum likelihood estimation
    theta0 = np.append(beta0, sigma0_list[1:])
    print("theta:", theta0)
    res = crowdsourcing_model(X, Y1_annotation, A1_annotation)
    print(res)
    # a = neg_loglikelihood(theta0, X, Y1_annotation, A1_annotation)
    # print(a)
    # theta0 = theta0 / 2
    # print("theta:", theta0)
    # b = neg_loglikelihood(theta0, X, Y1_annotation, A1_annotation)
    # print(b)


