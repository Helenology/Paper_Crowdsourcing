#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 21:21
# @Author  : Helenology
# @Site    : 
# @File    : synthetic_dataset.py
# @Software: PyCharm


import numpy as np


def construct_synthetic_dataset(N, p, beta0, seed=0):
    """
    Construct a synthetic dataset.
    :param beta0:
    :param N: the number of the whole unlabeled dataset
    :param p: the dimension of features
    :param seed: the random seed for reproducibility
    :return: the unlabeled dataset with X(features) and Y(true labels)
    """

    X = np.random.randn(N, p)  # features
    X[:, 0] = 1  # set the first columns of X to be constants
    Y_true = (X.dot(beta0) > 0).astype(int)  # true labels
    return X, Y_true


if __name__ == '__main__':
    N = 1000
    p = 10
    beta0 = np.ones(p)  # true parameter of interest
    X, Y_true = construct_synthetic_dataset(N, p, beta0, seed=0)
