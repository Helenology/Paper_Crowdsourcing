#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 10:22
# @Author  : Helenology
# @Site    :
# @File    : main_simulation.py
# @Software: PyCharm


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing as mp


import sys
import os
import time
import csv
sys.path.append(os.path.abspath('../data/'))
sys.path.append(os.path.abspath('../model/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
# from OS import *
from INR import *


def get_hyper_parameter(hyper_parameters, name):
    param = hyper_parameters[hyper_parameters['Hyper_Parameter'] == name].iloc[0, 1]
    return param


def map_func(params):
    np.set_printoptions(precision=3)
    # get hyper parameters
    path = "../Hyper_Parameters.xlsx"
    hyper_parameters = pd.read_excel(path)
    seed = get_hyper_parameter(hyper_parameters, "seed")  # random seed
    N = get_hyper_parameter(hyper_parameters, "N")  # the sample size of the whole unlabeled dataset
    p = get_hyper_parameter(hyper_parameters, "p")  # the dimension of the features
    M = get_hyper_parameter(hyper_parameters, "M")  # the number of the first-round annotators

    # get iteration parameters
    alpha = params[0]
    alpha_list = [alpha] * M
    subset_ratio = params[1]
    rep = params[2]
    print(f"Launching Process with alpha({alpha}), subset({subset_ratio}), and repetition({rep})")

    # parameter of interest + synthetic dataset
    np.random.seed(seed)  # set random seed
    beta_star = np.ones(p)  # the true parameter of interest
    X, Y_true = construct_synthetic_dataset(N, p, beta_star, seed=rep)  # generate synthetic dataset

    # generate synthetic annotators
    sigma_star = np.ones(M)
    sigma_star[1:int(M / 2)] *= 0.2  # np.arange(start=0.1, stop=10.1, step=(10 / M))[:(-1)]
    sigma_star[int(M / 2):M] *= 5

    # first-round sample selection
    np.random.seed(rep)
    X1, X2, Y1_true, Y2_true = train_test_split(X, Y_true, test_size=(1 - subset_ratio), random_state=rep)
    # synthetic annotations
    A1_annotation, Y1_annotation = synthetic_annotation(X1, beta_star, M, sigma_star, alpha_list, seed=rep)

    # One-Step Algorithm
    t1 = time.time()
    os = INR(X1, Y1_annotation, A1_annotation)
    _, os_iter = os.INR_algorithm(maxIter=1, mseWarn=1, epsilon=1e-6)
    t2 = time.time()
    os_time = t2 - t1
    os_beta_mse = mean_squared_error(beta_star, os.beta_hat)
    os_sigma_mse = mean_squared_error(sigma_star, os.sigma_hat)

    # INR Algorithm
    t3 = time.time()
    inr = INR(X1, Y1_annotation, A1_annotation)
    _, inr_iter = inr.INR_algorithm(maxIter=20, mseWarn=1, epsilon=1e-6)
    t4 = time.time()
    inr_time = t4 - t3
    inr_beta_mse = mean_squared_error(beta_star, inr.beta_hat)
    inr_sigma_mse = mean_squared_error(sigma_star, inr.sigma_hat)

    results = [alpha, subset_ratio, rep,
               os_time, os_iter, os_beta_mse, os_sigma_mse,
               inr_time, inr_iter, inr_beta_mse, inr_sigma_mse]

    print(f"Record Results from Process with alpha({alpha}), subset({subset_ratio}), and repetition({rep})")
    with open(f'./results/simulation-alpha({alpha})-subset({subset_ratio}).csv',
            'a') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(results)
    print(f"OS MSE:\n\tos_beta_mse: {os_beta_mse}\n\tos_sigma_mse: {os_sigma_mse:.6f}")
    print(f"INR MSE:\n\tinr_beta_mse: {inr_beta_mse}\n\tinr_sigma_mse: {inr_sigma_mse:.6f}")
    return None


if __name__ == "__main__":
    np.set_printoptions(precision=5)
    # get hyper parameters
    path = "../Hyper_Parameters.xlsx"
    hyper_parameters = pd.read_excel(path)
    alpha = get_hyper_parameter(hyper_parameters, "alpha")  # the instance assignment probability
    alphas = [float(item) for item in alpha.split()]  # convert to a list of alphas
    # alphas = [0.1]
    repetition = get_hyper_parameter(hyper_parameters, "repetition")  # the repetition times
    subset_ratio = get_hyper_parameter(hyper_parameters, "subset_ratio")
    subset_ratio_list = [float(item) for item in subset_ratio.split()]
    # subset_ratio_list = [0.1]
    # repetition = 100

    for alpha in alphas:
        for subset_ratio in subset_ratio_list:
            # create the csv file for results
            with open(
                    f'./results/simulation-alpha({alpha})-subset({subset_ratio}).csv',
                    'a') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(
                    ['alpha', 'subset_ratio', 'repetition',
                     'os_time', 'os_iter', 'os_beta_mse', 'os_sigma_mse',
                     'inr_time', 'inr_iter', 'inr_beta_mse', 'inr_sigma_mse'])

    # multi-processing
    NUM_THREADS = 4
    NUM_CPU = int(mp.cpu_count())
    print(f'CPU总数: {NUM_CPU}')
    os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
    NUM_PROCESS = NUM_CPU // NUM_THREADS
    print(f'最大并行进程数: {NUM_PROCESS}')
    # parameter dic for multi-processing
    # param_list = [[i, j, 58] for i in alphas for j in subset_ratio_list]
    param_list = [[i, j, k] for i in alphas for j in subset_ratio_list for k in range(repetition)]

    # multiprocessing
    with mp.Pool(NUM_PROCESS) as pool:
        results = pool.map(map_func, param_list)
