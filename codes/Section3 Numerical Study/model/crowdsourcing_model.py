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
from scipy.optimize import minimize, newton, approx_fprime
sys.path.append(os.path.abspath('../data/'))
from synthetic_annotators import *
import copy
from sklearn.metrics import mean_squared_error
import time


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
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20    # otherwise it could lead to inf
    Phi_matrix_minus = norm.cdf(-inner_matrix)     # 1 - Phi_matrix
    Phi_matrix_minus[Phi_matrix_minus < 1e-20] = 1e-20  # otherwise it could lead to inf

    log_likelihood = Y_annotation * np.log(Phi_matrix) + (1 - Y_annotation) * np.log(Phi_matrix_minus)
    log_likelihood = np.sum(log_likelihood * A_annotation)
    log_likelihood = log_likelihood / n / M
    return - log_likelihood


def compute_phi_Phi(inner_matrix):
    phi_matrix = norm.pdf(inner_matrix)
    Phi_matrix = norm.cdf(inner_matrix)
    Phi_matrix[Phi_matrix < 1e-20] = 1e-20
    Phi_matrix_minus = 1 - Phi_matrix
    Phi_matrix_minus[Phi_matrix_minus < 1e-20] = 1e-20
    return phi_matrix, Phi_matrix, Phi_matrix_minus


def rho_function(phi_matrix, Phi_matrix, Phi_matrix_minus):
    rho_matrix = phi_matrix / Phi_matrix
    rho_matrix_minus = phi_matrix / Phi_matrix_minus
    return rho_matrix, rho_matrix_minus


def score_function(theta, X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]

    # beta and sigma
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list).reshape((1, M))  # (1, M)
    score_vec = np.zeros(p + M - 1)

    # common value $Y_{im} \rho(u_{im}) - (1 - Y_{im}) \rho(-u_{im})$
    X_sigma = X.reshape((n, p, 1)) / all_sigma_list.reshape((1, 1, M))  # (n, p, M)
    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list  # (n, M)
    phi_matrix, Phi_matrix, Phi_matrix_minus = compute_phi_Phi(inner_matrix)
    rho_matrix, rho_matrix_minus = rho_function(phi_matrix, Phi_matrix, Phi_matrix_minus)
    common_value = Y_annotation * rho_matrix - (1 - Y_annotation) * rho_matrix_minus  # (n, M)

    # first-order derivative w.r.t. beta
    common_matrix = common_value * A_annotation  # (n, M)
    common_matrix = common_matrix.reshape((n, 1, M))  # (n, 1, M)
    score_beta = common_matrix * X_sigma  # (n, p, M)
    score_beta = score_beta.sum(axis=(0, 2))
    score_vec[0:p] = score_beta

    # first-order derivative w.r.t. sigma
    X_beta_sigma2 = X.dot(beta).reshape((n, 1)) / all_sigma_list**2  # (n, M)
    common_matrix = common_matrix.reshape((n, M))
    tmp_matrix = - X_beta_sigma2 * common_value * A_annotation
    score_sigma = tmp_matrix.sum(axis=0)  # (M)
    score_vec[p:] = score_sigma[1:]

    # adjust score vector
    score_vec = score_vec / n / M
    return -score_vec


def hessian_function(theta, X, Y_annotation, A_annotation):
    n, p = X.shape
    M = Y_annotation.shape[1]

    # beta and sigma
    beta = theta[0:p]
    sigma_list = theta[p:]
    all_sigma_list = np.append([1], sigma_list).reshape((1, M))  # (1, M)
    hessian_matrix = np.zeros((p+M-1, p+M-1))

    # common value the same in gradient $Y_{im} \rho(u_{im}) - (1 - Y_{im}) \rho(-u_{im})$
    inner_matrix = X.dot(beta).reshape((n, 1)) / all_sigma_list  # (n, M)
    phi_matrix, Phi_matrix, Phi_matrix_minus = compute_phi_Phi(inner_matrix)
    rho_matrix, rho_matrix_minus = rho_function(phi_matrix, Phi_matrix, Phi_matrix_minus)
    gradient_common_value = Y_annotation * rho_matrix - (1 - Y_annotation) * rho_matrix_minus  # (n, M)
    # common value in hessian
    hessian_tmp = (Y_annotation - Phi_matrix) * phi_matrix / Phi_matrix / Phi_matrix_minus
    hessian_common_value = -hessian_tmp * inner_matrix - hessian_tmp**2

    # matrix_11
    matrix_11 = A_annotation * hessian_common_value / all_sigma_list**2  # (n, M)
    diag = np.diag(matrix_11.sum(axis=1))  # (n, n)
    # record
    hessian_matrix[0:p, 0:p] = X.transpose().dot(diag).dot(X)

    # matrix_12 and matrix_21
    # first part
    a = -A_annotation * gradient_common_value / all_sigma_list**2  # (n, M)
    a = a.reshape((n, 1, M))  # (n, 1, M)
    first_part = a * X.reshape((n, p, 1))  # (n, p, M)
    # second part
    second_part = -A_annotation * hessian_common_value / all_sigma_list**3  # (n, M)
    X_beta = X.dot(beta).reshape((n, 1))  # (, 1)
    second_part = second_part * X_beta  # (n, M)
    second_part = second_part.reshape((n, 1, M))
    second_part = second_part * X.reshape((n, p, 1))
    matrix_12 = first_part + second_part  # (n, p, M)
    matrix_12 = matrix_12.sum(axis=0)  # (p, M)
    # record
    hessian_matrix[0:p, p:] = matrix_12[:, 1:]
    hessian_matrix[p:, 0:p] = matrix_12[:, 1:].transpose()

    # matrix_22
    first_part22 = A_annotation * gradient_common_value / all_sigma_list**3  # (n, M)
    first_part22 *= 2 * X_beta  # (n, 1)
    second_part22 = A_annotation * hessian_common_value / all_sigma_list**4
    second_part22 *= X_beta**2
    matrix_22 = first_part22 + second_part22  # (n, M)
    matrix_22 = matrix_22.sum(axis=0)  # (M, )
    hessian_matrix[p:, p:] = np.diag(matrix_22[1:])

    hessian_matrix /= n * M
    return -hessian_matrix


def crowdsourcing_model(X, Y_annotation, A_annotation, optimize):
    n, p = X.shape
    M = Y_annotation.shape[1]
    if optimize == -1:
        print(f"Use default scipy optimization with its point-wise estimated gradient.")
        theta_start = np.random.rand(p + M - 1)  # initial values
        res = minimize(neg_loglikelihood,
                       theta_start,
                       method='L-BFGS-B',  # L-BFGS-B, SLSQP, Newton-CG
                       args=(X, Y_annotation, A_annotation),
                       options={'disp': False})
        return res
    if optimize == 0:
        print(f"Use default scipy optimization.")
        theta_start = np.random.rand(p + M - 1)  # initial values
        res = minimize(neg_loglikelihood,
                       theta_start,
                       method='L-BFGS-B',  # L-BFGS-B, SLSQP, Newton-CG
                       args=(X, Y_annotation, A_annotation),
                       jac=score_function,
                       options={'disp': False})
        return res
    elif optimize == 1:
        print(f"Use Newton-Raphson algorithm.")
        theta_pre = None
        theta_now = np.random.rand(p + M - 1)
        maxIter = 200
        i = 0
        gtol = 1e-6
        while i < maxIter:
            i += 1
            print(f"===The {i}th Iteration===", end="")
            theta_pre = copy.copy(theta_now)
            gradient = score_function(theta_now, X, Y_annotation, A_annotation)
            hessian = hessian_function(theta_now, X, Y_annotation, A_annotation)
            theta_now = theta_now - np.linalg.inv(hessian).dot(gradient)

            change = np.linalg.norm(theta_now - theta_pre)
            print(f"change: {change:.6f}")
            if change <= gtol:
                break
        res = theta_now
        return res


if __name__ == '__main__':
    N = 10001
    p = 11
    M = 11
    alpha_list = [1] * M
    X = np.random.randn(N, p)  # features
    X[:, 0] = 1  # set the first columns of X to be constants
    beta0 = np.ones(p)  # np.random.rand(p)
    Y_true = (X.dot(beta0) > 0).astype(int)  # true labels
    sigma0_list = np.ones(M)
    seed = 0
    A1_annotation, Y1_annotation = synthetic_annotation(X, beta0, M, sigma0_list, alpha_list, 0, seed=seed)

    # maximum likelihood estimation
    theta0 = np.append(beta0, sigma0_list[1:])
    # hessian = hessian_function(theta0, X, Y1_annotation, A1_annotation)
    # print(f"hessian: {hessian}")
    # print(hessian.shape)
    t1 = time.time()
    res = crowdsourcing_model(X, Y1_annotation, A1_annotation, optimize=1)
    print(res)
    # rmse = np.sqrt(mean_squared_error(res, theta0))
    # print(f"RMSE:{rmse:.6f}")
    t2 = time.time()
    print(f"time: {t2 - t1:.6f}")
