# Crowdsourcing

Relevant codes and some illustrations for the working paper *"A Statistical Framework for Hybrid
Human-Machine Annotation"*.

## A Brief Taste

- [ ] todo

## Codes

- `0611 Update`: 搭建了大致的代码结构，学习了argparse的使用，构建模拟数据集


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