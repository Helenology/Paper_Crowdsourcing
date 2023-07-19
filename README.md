# Crowdsourcing

Relevant codes and some illustrations for the working paper *"A Statistical Framework for Hybrid
Human-Machine Annotation"*.

## A Brief Taste

- [ ] todo

## Coding History

- `0611 Update`: 搭建了大致的代码结构，学习了argparse的使用，构建模拟数据集
- `0612 Update`: 第一阶段待标注数据集抽取、标注者随机分配第一阶段数据集，模拟标注数据生成
- `0613 Update`: 模型算法
- `0614`: debug
- `0615`: 修改算法优化为 Netwon-Raphson，测试
- `0616`: 发现 Netwon-Raphson 一阶导代入 python 自带模块后报错，还在 debug
- `0616-0618`: 全在debug一阶导，然后0618晚上发现在某个地方转换一下正负号就对了，然后顺着这个思路去检查，发现了bug！
- `0619`: debug二阶导，有些参数迭代后发散，于是猜测代码或者pdf里写错了，保证代码和pdf一致后，一定是pdf哪里错了，于是顺着这个思路去找，找到了一个地方的数学推导有问题，改正过来以后结果没问题了
- `0623`: 文章框架有大的改动，暂停更新
- `0706`: 更新INR算法，给定 $\sigma_m$ 真值优化 $\beta$ 没有问题，目前的bug是给定 $\beta$ 真值优化 $\sigma_m$ 仍然有bug，正在debug
- `0707`: 检查了公式，发现没有问题，因此是算法比较依赖于初始值，如果初始值在真值附近，那么优化会比较容易，如果初始值选的不好， $\sigma_m$ 可能出现负数，或者发散，因此接下来考虑利用这个观察和加入格点搜索来优化
- `0708`: 对于 $\sigma_m$ 增加了 reinitialization 部分的代码，如果给定 $\beta$ 的真值，优化没有太大的问题了，但是没给定真值时仍然有点 bug
- `0709`: 整理完 INR代码+ One-Step代码
- `0710`: 喝酒去了
- `0711`: 部署完并行计算的代码了，目前有个小问题，那就是主程序也许使用共享内存可以加快速度，但是我简单尝试了一下本地共享内存会报错，为了跳过这个bug，只好使用一种较为stupid的办法，那就是控制随机数后在各个进程中重新生成一模一样的数据
- `0712`: debug（INR算法出现了很多次崩溃）
  > 1. Beta的初始值都用annotator1来估计，所以INR的beta初始值变好了
  > 2. INR每一步迭代，先更新sigma，再更新beta
  > 	【原因】beta的初始值很好，先固定beta更新sigma，可以让sigma迭代到一个较好的值时再更新beta
  > 3. INR每一步迭代，更新sigma时更新好几次（我这里选择的5次），再更新一次beta
  	【原因】观察发现，更新一次sigma时sigma的值仍然不太好，此时更新beta会让beta变差，于是最终的结果变得很差
- `0719`: INR存在异常值，bug主要有两个：
  1. $\widehat sigma_m$ 在迭代中出现了负数，解决办法：使用调整初始值的方式重新初始化负数的 $\widehat sigma_m$
  2. 之前写的基于随机初始化 $\widehat sigma_m$ 的代码中有一个seldomly changed case，排查发现由于此时初始值给的很好，所以经常是seldomly change case，因此删去此部分代码

## Codes

All the relevant codes and the corresponding graphs are listed in [codes](./codes/). A tree of the belongings are as follows

```
codes
├── Section2.2 Annotator Selection
│   ├── plot_trace.R
│   ├── sigma_simulation.ipynb
│   ├── trace.csv
│   └── trace.pdf
├── Section3 Numerical Study
│   ├── Hyper_Parameters.xlsx
│   ├── __pycache__
│   ├── data
│   ├── main
│   ├── model
│   └── utils.py
└── 验证加入超级annotator之后是否结果变差
    ├── MLE.ipynb
    ├── test.csv
    └── 加入专家对比.R

8 directories, 9 files
```