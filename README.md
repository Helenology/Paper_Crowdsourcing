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