#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 22:42
# @Author  : Helenology
# @Site    : 
# @File    : synthetic_annotators.py
# @Software: PyCharm

import numpy as np


def synthetic_annotation(X1, beta0, M, sigma0_list, alpha_list, per_min):
    """

    :param X1: first-round feature vectors
    :param beta0: parameter of interest
    :param M: the number of annotators
    :param sigma0_list: the standard deviation of first-round annotators
    :param alpha_list: the probability of assigning each instance to each annotator
    :param per_min: the minimum number of annotators assigned to each instance
    :return: An annotation assignment matrix of size=(n, M) and a first-round annotation matrix of size=(n, M)
    """
    n = X1.shape[0]  # size of first-round selected samples
    #################### annotation assignment matrix
    A1_annotation = np.random.binomial(1, alpha_list, size=(n, M))
    # the row summation of A1_annotation should not be 0
    A1_rowsum = A1_annotation.sum(axis=1)
    if (A1_rowsum < per_min).sum() > 0:
        print(f"Warning: The per_min({per_min}) is not reached! Please modify this code later!")
    i = 0
    while (A1_rowsum == 0).sum() > 0:
        A1_annotation = np.random.binomial(1, alpha_list, size=(n, M))
        A1_rowsum = A1_annotation.sum(axis=1)
        print(f"A1_rowsum has element 0. Try {i} times.")
        i = i + 1

    #################### first-round annotation matrix
    epsilon = np.random.normal(loc=0, scale=sigma0_list, size=(n, M))  # random noise
    common_decision = (X1.dot(beta0)).reshape((n, 1))
    Y1_annotation = (common_decision + epsilon > 0).astype(int)  # first-round annotation matrix
    Y1_annotation *= A1_annotation  # only preserve where a_im=1

    return A1_annotation, Y1_annotation


