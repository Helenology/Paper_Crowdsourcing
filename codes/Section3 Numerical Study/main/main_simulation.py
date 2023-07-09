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

sys.path.append(os.path.abspath('../data/'))
sys.path.append(os.path.abspath('../model/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
from crowdsourcing_model import *
from OS import *
from INR import *
import csv


def get_hyper_parameter(name):
    param = hyper_parameters[hyper_parameters['Hyper_Parameter'] == name].iloc[0, 1]
    return param


def main():
    pass


def map_func():
    pass

if __name__ == "__main__":
    hyper_parameters = pd.read_excel(
        "/Users/helenology/Desktop/光华/ 论文/4-Crowdsourcing/codes/Section3 Numerical Study/Hyper_Parameters.xlsx")

    # hyper parameters
    seed = get_hyper_parameter("seed")  # random seed
    N = get_hyper_parameter("N")  # the sample size of the whole unlabeled dataset
    p = get_hyper_parameter("p")  # the dimension of the features
    subset_ratio = get_hyper_parameter("subset_ratio")  # subset_ratio = |first-round subset| / N
    subset_ratio_list = [float(item) for item in subset_ratio.split()]  # convert to a list of subset ratios
    M = get_hyper_parameter("M")  # the number of the first-round annotators
    alpha = get_hyper_parameter("alpha")  # the instance assignment probability
    alphas = [float(item) for item in alpha.split()]  # convert to a list of alphas
    repetition = get_hyper_parameter("repetition")  # the repetition times
    alphas = [0.2]
    subset_ratio_list = [0.1]


    ##################  parameter of interest + synthetic dataset
    np.random.seed(seed)  # set random seed
    beta_star = np.ones(p)  # the true parameter of interest
    X, Y_true = construct_synthetic_dataset(N, p, beta_star, seed=0)  # generate synthetic dataset

    # generate synthetic annotators
    sigma_star = np.ones(M)
    sigma_star[1:] *= np.arange(start=0.1, stop=10.1, step=(10 / M))[:(-1)]

    # parameter set
    theta_star = np.append(beta_star, sigma_star[1:])  # true parameters

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
    param_dict = {}
    task_num = 1







    # create the csv file for results
    with open('/Users/helenology/Desktop/光华/ 论文/4-Crowdsourcing/codes/Section3 Numerical Study/main/results/simulation.csv', 'a') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['alpha', 'subset_ratio', 'repetition', 'os_time', 'os_beta_mse', 'os_sigma_mse', 'inr_time', 'inr_beta_mse', 'inr_sigma_mse'])

    for alpha in alphas:
        alpha_list = np.ones(M) * alpha
        for subset_ratio in subset_ratio_list:
            task_name = f"task{task_num}"
            param_dict[task_name] = [alpha, subset_ratio]
            task_num += 1

            for rep in range(repetition):
                print(f"{task_name} repetition {rep}")
                np.random.seed(rep)
                X1, X2, Y1_true, Y2_true = train_test_split(X, Y_true, random_state=rep,
                                                            test_size=(1 - subset_ratio))
                # synthetic annotations
                A1_annotation, Y1_annotation = synthetic_annotation(X1, beta_star, M, sigma_star, alpha_list, seed=rep)

                # One-Step Algorithm
                t1 = time.time()
                os = OS(X1, Y1_annotation, A1_annotation)
                os.OS_algorithm()
                t2 = time.time()
                os_time = t2 - t1
                os_beta_mse = mean_squared_error(beta_star, os.beta_hat)
                os_sigma_mse = mean_squared_error(sigma_star, os.sigma_hat)

                # INR Algorithm
                t3 = time.time()
                inr = INR(X1, Y1_annotation, A1_annotation)
                inr.INR_algorithm(maxIter=50)
                t4 = time.time()
                inr_time = t4 - t3
                inr_beta_mse = mean_squared_error(beta_star, inr.beta_hat)
                inr_sigma_mse = mean_squared_error(sigma_star, inr.sigma_hat)

                results = [alpha, subset_ratio, rep, os_time, os_beta_mse, os_sigma_mse, inr_time, inr_beta_mse, inr_sigma_mse]
                with open(
                        '/Users/helenology/Desktop/光华/ 论文/4-Crowdsourcing/codes/Section3 Numerical Study/main/results/simulation.csv',
                        'a') as f:
                    csv_write = csv.writer(f)
                    csv_write.writerow(results)








