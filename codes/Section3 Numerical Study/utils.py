#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/5 21:25
# @Author  : Helenology
# @Site    : 
# @File    : utils.py
# @Software: PyCharm


import numpy as np
from scipy.stats import norm


def phi_function(x):
    return norm.pdf(x)


def phi_dot_function(x):
    return -x * phi_function(x)


def Phi_function(x):
    return norm.cdf(x)


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
