#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 22:02
# @Author  : Helenology
# @Site    :
# @File    : simulation.py
# @Software: PyCharm

import sys
import os
import time

sys.path.append(os.path.abspath('../data/'))
sys.path.append(os.path.abspath('../model/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
from crowdsourcing_model import *
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing


def main():
    try:
        # seed = args.seed  # random seed
        # N = args.sample_size  # the sample size of the whole unlabeled dataset
        # p = args.dimension  # the dimension of the features
        # first_round_ratio = args.first_round_ratio  # the size of first-round subset
        # M = args.annotator_number  # the number of the first-round annotators
        # alpha_list = np.array(
        #     args.alpha_list)  # the probability of assigning first-round instances to first-round annotators
        # per_min = args.per_min  # the minimum number of annotators assigned to each instance
        # repetition = args.repetition  # the repetition time
        # optimize = args.optimize  # use the default scipy optimization
    #         print(f"Received hyper-parameters:")
    #         print(f"\t- Random Seed: {seed}")
    #         print(f"\t- Sample Size: N={N}")
    #         print(f"\t- Dimension: p={p}")
    #         print(f"\t- First-Round Ratio: {first_round_ratio}")
    #         print(f"\t- First-Round Annotators: M={M}")
    #         print(f"\t- First-Round alpha_list: {alpha_list}")
    except Exception as e:  # errors
        print(e)

    # raise error part
    # sample size (N) should be larger than dimension (p)
    if N <= p:
        raise ValueError(f"Sample size({N}) is not larger than dimension({p})!")
    # first_round_ratio should be \in (0, 1)
    if first_round_ratio <= 0 or first_round_ratio >= 1:
        raise ValueError(f"first_round_ratio({first_round_ratio}) is <= 0 or >= 1!")
    # alpha_m should be \in (0, 1) and summation > 0
    if len(alpha_list) == 0 or alpha_list.sum() == 0:
        raise ValueError(f"alpha_list:{alpha_list} has no elements or its summation is 0!")
    if (alpha_list < 0).sum() + (alpha_list > 1).sum() > 0:
        raise ValueError(f"There exists at least one alpha < 0 or > 1!")

    # trim the alpha_list / M
    if len(alpha_list) == 1:
        alpha_list = np.array([alpha_list[0]] * M)
    elif len(alpha_list) != M:
        print(f"Warning: The number of first-round annotators is M={M}.")
        tmplen = min(len(alpha_list), 5)
        print(f"Warning: alpha_list[:{tmplen}]={alpha_list[0:tmplen]} with size={len(alpha_list)}.")
        if len(alpha_list) < M:
            print(f"Warning: We change M from ({M}) to ({len(alpha_list)}).")
            M = len(alpha_list)
        elif len(alpha_list) > M:
            print(f"Warning: We only reserve the first M({M}) elements.")
            alpha_list = alpha_list[0:M]
    #     print(f"\t- Average # of Annotators Per Instance: {alpha_list.sum()}")
    #     print(f"\t- Min Average # of Annotators Per Instance: {per_min}")

    ##################  parameter of interest + synthetic dataset
    np.random.seed(0)
    beta0 = np.ones(p)  # the true parameter of interest
    X, Y_true = construct_synthetic_dataset(N, p, beta0, seed=0)  # generate synthetic dataset

    # generate synthetic annotators
    sigma0_list = np.ones(M)
    # sigma0_list[1:] *= np.arange(start=0.1, stop=10.1, step=(10 / M))[1:]  # np.random.chisquare(1, size=M-1)
    sigma0_list[sigma0_list < 1e-3] = 1e-3  # sigma should not be 0
    theta0 = np.append(beta0, sigma0_list[1:])  # true parameters

    rmse_results = []
    old_rmse = None

    for _ in range(repetition):
        np.random.seed(seed)
        # first-round sample selection
        X1, X2, Y1_true, Y2_true = train_test_split(X, Y_true, random_state=seed, test_size=(1 - first_round_ratio))

        # synthetic annotations
        A1_annotation, Y1_annotation = synthetic_annotation(X1, beta0, M, sigma0_list, alpha_list, per_min, seed=seed)

        # maximum likelihood estimation
        t1 = time.time()
        if optimize == 0:
            iter = 0
            maxTry = 2
            rmse_result = [seed, None, optimize]  # seed, RMSE, optimize_method

            np.random.seed(seed)
            while iter < maxTry:
                res = crowdsourcing_model(X1, Y1_annotation, A1_annotation, optimize=0)
                if res.success:
                    rmse = np.sqrt(mean_squared_error(res.x, theta0))
                    if old_rmse is None or (rmse < 5 * old_rmse or rmse > old_rmse / 5):
                        t2 = time.time()
                        print(f"Success: seed={seed} with RMSE{rmse} and time{t2 - t1: .6f}")
                        rmse_result[1] = rmse
                        break
                iter += 1

            if rmse_result[1] is None:
                print(f"Use default scipy optimization with its point-wise estimated gradient.")
                res = crowdsourcing_model(X1, Y1_annotation, A1_annotation, optimize=-1)  # optimize again
                if res.success:
                    rmse = np.sqrt(mean_squared_error(res.x, theta0))
                    if old_rmse is None or (rmse < 5 * old_rmse or rmse > old_rmse / 5):
                        t2 = time.time()
                        print(f"Success: seed={seed} with RMSE{rmse} and time{t2 - t1: .6f}")
                        rmse_result[1] = rmse
                        rmse_result[2] = -1
            if rmse_result[1] is None:
                print(f"Error: seed={seed} failed in optimization!")
                rmse_result[1] = "ERROR"

            rmse_results.append(rmse_result)

        elif optimize == 1:
            res = crowdsourcing_model(X1, Y1_annotation, A1_annotation, optimize=optimize)  # optimize again
            rmse = np.sqrt(mean_squared_error(res, theta0))
            rmse_results.append([seed, rmse, 1])
            t2 = time.time()
            print(f"Seed={seed} with RMSE{rmse} and time{t2 - t1: .6f}")

        print(res)
        # rmse_DF = pd.DataFrame(rmse_results, columns=['seed', 'RMSE', 'opt'])
        # rmse_DF.to_csv(f"./results/rmse-alpha{alpha_list[0]}-r{first_round_ratio}.csv")
        seed += 1


if __name__ == "__main__":
    main()
