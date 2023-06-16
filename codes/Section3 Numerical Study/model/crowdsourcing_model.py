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
from scipy.optimize import minimize, newton, approx_fprime
from scipy.stats import norm
sys.path.append(os.path.abspath('../data/'))
from synthetic_annotators import *


def neg_loglikelihood(theta, X, Y_annotation, A_annotation):
    """
    Compute the negative log-likelihood.
    :param X: features (n x p)
    :param Y_annotation: Annotations (n x M)
    :param A_annotation: Binary indicators (n x M)
    :param theta: parameters of size (p+M-1)
    :return: negative log-likelihood
    """
    n, p = X.shape
    M = Y_annotation.shape[1]

    # beta and sigma_list
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list).reshape((1, M))  # (1, M)

    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list
    Phi_matrix = norm.cdf(inner_matrix)
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20  # otherwise it could lead to inf
    Phi_matrix1 = norm.cdf(-inner_matrix)  # 1 - Phi_matrix
    Phi_matrix1[Phi_matrix1 < 1e-20] = 1e-20  # otherwise it could lead to inf

    log_likelihood = Y_annotation * np.log(Phi_matrix) + (1 - Y_annotation) * np.log(Phi_matrix1)
    log_likelihood = np.sum(log_likelihood * A_annotation)
    log_likelihood = log_likelihood / n / M
    return - log_likelihood


def rho_function(inner_matrix):
    Phi_matrix = norm.cdf(inner_matrix)
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20  # otherwise it could lead to inf
    tmp = norm.pdf(inner_matrix) / Phi_matrix
    return tmp


def score_function(theta, X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]
    # beta and sigma
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list).reshape((1, M))  # (1, M)
    score_vec = np.zeros(p + M - 1)

    # first-order derivative w.r.t. beta
    X_sigma = np.dot(X.reshape((n, p, 1)), 1 / all_sigma_list)  # (n, p, M)
    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list  # (n, M)
    rho_matrix = rho_function(inner_matrix)
    rho_matrix_minus = rho_function(-inner_matrix)
    common_value = Y_annotation * rho_matrix - (1 - Y_annotation) * rho_matrix_minus
    common_matrix = common_value * A_annotation  # (n, M)
    score_beta = np.tensordot(common_matrix, X_sigma, axes=[[0, 1], [0, 2]])
    score_vec[0:p] = score_beta

    # first-order derivative w.r.t. sigma
    # X_beta_sigma = X.dot(beta).reshape((n, 1)) / all_sigma_list**2  # (n, M)
    # tmp_matrix = - X_beta_sigma * common_matrix
    # score_sigma = tmp_matrix.sum(axis=0)  # (M)
    # score_vec[p:] = score_sigma[1:]  # (M-1)
    common_matrix = common_matrix[:, 0:(M-1)]  # (n, M-1)
    X_beta_sigma = X.dot(beta).reshape((n, 1)) / sigma_list**2  # (n, M-1)
    tmp_matrix = - X_beta_sigma * common_matrix  # (n, M-1)
    score_sigma = tmp_matrix.sum(axis=0)  # (M-1,)
    score_vec[p:] = score_sigma

    # adjust score vector
    score_vec = score_vec / M / n
    return score_vec


def hessian_function(theta, X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]
    # beta and sigma
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list).reshape((1, M))  # (1, M)
    hessian_matrix = np.zeros((p+M-1, p+M-1))

    # matrix_11
    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list  # (n, M)
    Phi_matrix = norm.cdf(inner_matrix)
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20  # otherwise it could lead to inf
    Phi_matrix1 = 1 - Phi_matrix
    Phi_matrix1[Phi_matrix1 < 1e-20] = 1e-20
    tmp = (Y_annotation - Phi_matrix) / Phi_matrix / Phi_matrix1
    tmp1 = (tmp * norm.pdf(inner_matrix) * (-inner_matrix) - tmp**2 * norm.pdf(inner_matrix)**2)
    matrix_11 = A_annotation * tmp1 / all_sigma_list**2
    diag = np.diag(matrix_11.sum(axis=1))  # (n,)
    matrix_11 = X.transpose().dot(diag).dot(X)
    hessian_matrix[0:p, 0:p] = matrix_11

    # matrix_12 and matrix_21
    rho_matrix = rho_function(inner_matrix)
    rho_matrix_minus = rho_function(-inner_matrix)
    common_value = Y_annotation * rho_matrix - (1 - Y_annotation) * rho_matrix_minus
    X_sigma = np.dot(X.reshape((n, p, 1)), 1 / all_sigma_list)  # (n, p, M)
    a = A_annotation * common_value  # (n, M)
    a = a.reshape((n, 1, M))  # (n, 1, M)
    matrix_12 = - a * X_sigma  # (n, p, M)
    tmp1  # (n, M)

    matrix_12 = matrix_12.sum(axis=0)  # (p, M)
    matrix_12 = matrix_12[:, 1:]
    hessian_matrix[0:p, p:] = matrix_12
    hessian_matrix[p:, 0:p] = matrix_12.transpose()

    # matrix_22

    return matrix_12



def crowdsourcing_model(X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]
    # theta_start = np.random.rand(p + M - 1)
    theta_start = np.ones(p + M - 1)
    # using built-in module needs about 2 seconds
    # Newton-CG
    res = minimize(neg_loglikelihood, theta_start,
                   method='L-BFGS-B',  # L-BFGS-B, SLSQP, Newton-CG
                   args=(X, Y_annotation, A_annotation),
                   jac=score_function,
                   options={'disp': True})
    return res


if __name__ == '__main__':
    N = 1001
    p = 11
    M = 21
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

    h = hessian_function(theta0, X, Y1_annotation, A1_annotation)
    # print(h)
    print(h.shape)
    # res = crowdsourcing_model(X, Y1_annotation, A1_annotation)
    # print(res)
    # a = neg_loglikelihood(theta0, X, Y1_annotation, A1_annotation)
    # print(a)
    # theta0 = theta0 / 2
    # print("theta:", theta0)
    # b = neg_loglikelihood(theta0, X, Y1_annotation, A1_annotation)
    # print(b)


