#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/8/18 23:26
# @Author  : Helenology
# @Site    : 
# @File    : MV.py
# @Software: PyCharm

import numpy as np
from scipy import stats


def majority_voting(matrix):
    # 初始化存储众数和众数出现次数的数组
    modes = []
    counts = []

    for row in matrix:
        # 忽略值为-1的元素
        filtered_row = row[row != -1]

        # 如果过滤后的行不为空，求众数
        if len(filtered_row) > 0:
            mode_result = stats.mode(filtered_row)
            modes.append(mode_result.mode[0])
            counts.append(mode_result.count[0])
        else:
            modes.append(np.nan)  # 如果整行都是-1，可以放置NaN或其他标志
            counts.append(0)

    return np.array(modes)