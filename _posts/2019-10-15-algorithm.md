---
layout: post
title: "排料算法整理和设计"
subtitle: 'Slove 2D bin packing'
author: "sean"
header-img: "img/post-bg-unix-linux.jpg"
tags:
  - 算法

---



- [ ]  传统排料算法怎么做的：Algorithm for 2D irregular-shaped nesting problem based on the NFP algorithm and lowest-gravity-center principle
- [ ]  阿里的输入与输出具体什么情况：通过传统算法进行排列，并选择最好的几个结果进行评估，然后可以训练出一个序列学习的模型，可以给出最优的加入策略
- [ ]  阿里算法流程：输入箱子的参数，直接给出一个排序序列，按照这个序列把箱子给放进去，通过启发式算法（贪婪策略）进行箱子的排列
- [ ]  **算法折中策略**：（1）对于输出的序列他们采用的是对输出的序列进行排序 （2）即使是几个箱子，算法采用的也是贪婪算法，更小的局部最优 （3）本质上对位置没有预测
- [ ]  数据集的问题：http://euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507



**整体方案**：（1）采用[ESICUP](https://www.euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507)数据集进行排样（2）对给定的图像根据重心最低原则处理，然后通过算法对其形状及大小用向量进行评估（3）通过启发式算法排样，选择最优的多个排样结果输出，包括重心位置/配件方向及利用率（4）对初始输入的参数和输出的结果进行学习，建立映射关系（配件的方向和位置）（5）对初始解进行从下到上的优化，获得最优解

**细节调整**：（1）启发式算法现在只适应水平底部和给定边框（2）配件需要用重心策略进行方向调整后才能进行评估（3）欧式距离小于一定的值可以判定为配件类似

**重点算法**：（1）启发式排样策略（2）配件方向初始化（3）配件的评估参数（4）学习网络怎么实现（没有找到对应网络）

**传统组合优化+机器学习**：（1）阿里巴巴排箱算法，采用Pointer Network（Seq2Seq+Attention），但是只预测添加顺序，对具体排样结果不做预测，可以说是将二维转为一维，也可以说是没办法只能这么做了（2）Georgia Tech的提出的Embedding Graph的算法，主要面向是Travle salesmen这种问题，只面向于基于图的算法 

**算法优势**：（1）可以通过内容直接预测最终的大致位置与方向，以及整体排料效果（2）可以大大提高排样的速度（3）可能可以提高排样效果（4）可以为大规模排样分解做铺垫工作

**难点**：很多难点......

**解决方案**：一步步来

**部分备注**：（1）先从小规模训练起（2）可以提前采用监督学习，再采用强化学习，神经网络是BP网络

**潜在问题**：因为效果的差异性，可能最后的结果不一定合理（暂时不考虑）



### 论文

- [ ] Introduction

  - [ ] Background
- [ ] Literature Review

  - [ ] 2D Irregular Strips Packing
    - [ ] 
  - [ ] Neural Network in Combination Optimization
    - [ ] 
- [ ] The Difinition of the Problem
- [ ] Algorithm Outline
- [ ] Initialization and Classification Model

  - [ ] Ajdust strips using lowest-gravity-center principle
    - [ ] 参考 Algorithm for 2D irregular-shaped nesting problem based on the NFP algorithm and lowest-gravity-center principle
  - [ ] Clustering by features
    - [ ] 参考 Automatic Data Clustering Analysis of Arbitrary Shape with K-means and Enhanced Ant-based Template Mechanism或者直接处理
- [ ] Neural Network Method and Heuristic Optimization

  - [ ] Architecture of the network
    - [ ] 可能有可以将features对应到结果的学习网络
    - [ ] 直接尝试BP网络，但是具体的设计需要研究
    - [ ] 尽量实现，否则Ptr的效果很差没有创新

  - [ ] Heruistic Optimization Algorithm
    - [ ] 同调整算法
- [ ] Computational Results
- [ ] Conclusion
  - [ ] 取得的效果是可观的（理想化）
  - [ ] 可以转移到其他的问题，其量化的实现思想
- [ ] Acknowledge/References



![image-20191021005156381](https://tva1.sinaimg.cn/large/006y8mN6gy1g855gklphej31ju0rkwl6.jpg)

