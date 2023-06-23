# Crowdsourcing

Relevant codes and some illustrations for the working paper *"A Statistical Framework for Hybrid
Human-Machine Annotation"*.

## A Brief Taste

- [ ] todo

## Codes

- `0611 Update`: 搭建了大致的代码结构，学习了argparse的使用，构建模拟数据集
- `0612 Update`: 第一阶段待标注数据集抽取、标注者随机分配第一阶段数据集，模拟标注数据生成
- `0613 Update`: 模型算法
- `0614`: debug
- `0615`: 修改算法优化为 Netwon-Raphson，测试
- `0616`: 发现 Netwon-Raphson 一阶导代入 python 自带模块后报错，还在 debug
- `0616-0618`: 全在debug一阶导，然后0618晚上发现在某个地方转换一下正负号就对了，然后顺着这个思路去检查，发现了bug！
- `0619`: debug二阶导，有些参数迭代后发散，于是猜测代码或者pdf里写错了，保证代码和pdf一致后，一定是pdf哪里错了，于是顺着这个思路去找，找到了一个地方的数学推导有问题，改正过来以后结果没问题了
- `0623`: 文章框架有大的改动，暂停更新


- [ ] per_min那里，大致是希望如果实际不满足per_min个标注者，可不可以再分配几个标注者达到per_min的数量，是符合实际要求，但是不符合模型设定

All the relevant codes and the corresponding graphs are listed in [codes](./codes/). A tree of the belongings are as follows

```
codes
├── Section2.2 Annotator Selection
│   ├── plot_trace.R
│   ├── sigma_simulation.ipynb
│   ├── trace.csv
│   └── trace.pdf
├── Section2.4 Reinforced Labeling
│   ├── plot_round12_diff.R
│   ├── second-round probability.ipynb
│   ├── two-rounds-confidence.pdf
│   └── two_rounds_props.csv
├── Section3 Numerical Study
│   ├── Simulation Study.ipynb
│   ├── data
│   │   ├── __pycache__
│   │   │   └── synthetic_dataset.cpython-39.pyc
│   │   ├── synthetic_annotators.py
│   │   └── synthetic_dataset.py
│   ├── main
│   │   ├── __pycache__
│   │   │   └── simulation.cpython-39-pytest-6.2.4.pyc
│   │   └── simulation.py
│   └── utils
│       └── __pycache__
└── 验证加入超级annotator之后是否结果变差
    ├── MLE.ipynb
    ├── test.csv
    └── 加入专家对比.R

11 directories, 17 files