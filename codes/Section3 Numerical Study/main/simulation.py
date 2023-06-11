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
import argparse
import numpy as np
import pandas as pd


# receive arguments from command line or use the default values
parser = argparse.ArgumentParser(description='Argparse')
parser.add_argument('--sample_size', '-N', help='sample size', default=10000)
parser.add_argument('--dimension', '-p', help='dimension', default=10)
parser.add_argument('--annotator_number', '-M', help='the number of first-round annotators', default=100)
args = parser.parse_args()


def main():
    try:
        N = args.sample_size       # the sample size of the whole unlabeled dataset
        p = args.dimension         # the dimension of the features
        M = args.annotator_number  # the number of the first-round annotators
    except Exception as e:    # errors
        print(e)

    # generate the true parameter of interest
    beta0 = np.ones(p)

    # generate synthetic dataset
    X, Y_true = construct_synthetic_dataset(N, p, beta0, seed=0)

    # generate synthetic annotators
    construct_synthetic_annotators(M)


if __name__ == "__main__":
    main()
