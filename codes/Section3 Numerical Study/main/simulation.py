#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/11 22:02
# @Author  : Helenology
# @Site    : 
# @File    : simulation.py
# @Software: PyCharm

import sys
import os

sys.path.append(os.path.abspath('../data/'))
from synthetic_dataset import *  # codes to generate a synthetic dataset
from synthetic_annotators import *
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# receive arguments from command line or use the default values
parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--seed', '-seed', help='random seed', default=0, type=int)
parser.add_argument('--sample_size', '-N', help='sample size', default=10000, type=int)
parser.add_argument('--dimension', '-p', help='dimension', default=10, type=int)
parser.add_argument('--annotator_number', '-M', help='the number of first-round annotators', default=50, type=int)
parser.add_argument('--first_round_ratio', '-r', help='the ratio of first-round subset', default=0.1, type=float)
parser.add_argument('--alpha_list', '-alphas', nargs="*", type=float, help='probability of assigning instances',
                    default=[0.1])
parser.add_argument('--per_min', '-per_min', type=int, help='the minimum number of annotators assigned to each instance', default=0)
args = parser.parse_args()


def main():
    try:
        seed = args.seed      # random seed
        N = args.sample_size  # the sample size of the whole unlabeled dataset
        p = args.dimension    # the dimension of the features
        first_round_ratio = args.first_round_ratio  # the size of first-round subset
        M = args.annotator_number  # the number of the first-round annotators
        alpha_list = np.array(args.alpha_list)  # the probability of assigning first-round instances to first-round annotators
        per_min = args.per_min  # the minimum number of annotators assigned to each instance
        print(f"Received hyper-parameters:")
        print(f"\t- Random Seed: {seed}")
        print(f"\t- Sample Size: N={N}")
        print(f"\t- Dimension: p={p}")
        print(f"\t- First-Round Ratio: {first_round_ratio}")
        print(f"\t- First-Round Annotators: M={M}")
        print(f"\t- First-Round alpha_list: {alpha_list}")
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
    if len(alpha_list) != M:
        print(f"Warning: The number of first-round annotators is M={M}.")
        tmplen = min(len(alpha_list), 5)
        print(f"Warning: alpha_list[:{tmplen}]={alpha_list[0:tmplen]} with size={len(alpha_list)}.")
        if len(alpha_list) < M:
            print(f"Warning: We change M from ({M}) to ({len(alpha_list)}).")
            M = len(alpha_list)
        elif len(alpha_list) > M:
            print(f"Warning: We only reserve the first M({M}) elements.")
            alpha_list = alpha_list[0:M]
    print(f"\t- Average # of Annotators Per Instance: {alpha_list.sum()}")
    print(f"\t- Min Average # of Annotators Per Instance: {per_min}")

    ##################  parameter of interest + synthetic dataset
    beta0 = np.ones(p)  # the true parameter of interest
    X, Y_true = construct_synthetic_dataset(N, p, beta0, seed=seed)  # generate synthetic dataset

    ##################  first-round sample selection
    X1, X2, Y1_true, Y2_true = train_test_split(X, Y_true, random_state=0, test_size=(1 - first_round_ratio))
    n = X1.shape[0]     # the size of the first-round selected samples

    # generate synthetic annotators
    sigma0_list = np.ones(M)
    sigma0_list[1:] *= np.random.chisquare(1, size=M-1)
    sigma0_list[sigma0_list == 0] = 1e-3  # sigma should not be 0
    # synthetic annotations
    A1_annotation, Y1_annotation = synthetic_annotation(X1, beta0, M, sigma0_list, alpha_list, per_min)




if __name__ == "__main__":
    main()
