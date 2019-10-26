---
layout: post
title: "排料算法整理和设计"
subtitle: 'Slove 2D bin packing'
author: "sean"
header-img: "img/post-bg-unix-linux.jpg"
tags:
  - 算法

---



### 介绍部分

- [ ]  传统排料算法怎么做的：Algorithm for 2D irregular-shaped nesting problem based on the NFP algorithm and lowest-gravity-center principle
- [ ]  阿里的输入与输出具体什么情况：通过传统算法进行排列，并选择最好的几个结果进行评估，然后可以训练出一个序列学习的模型，可以给出最优的加入策略
- [ ]  阿里算法流程：输入箱子的参数，直接给出一个排序序列，按照这个序列把箱子给放进去，通过启发式算法（贪婪策略）进行箱子的排列
- [ ]  **算法折中策略**：（1）对于输出的序列他们采用的是对输出的序列进行排序 （2）即使是几个箱子，算法采用的也是贪婪算法，更小的局部最优 （3）本质上对位置没有预测
- [ ]  数据集的问题：http://euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507



### 整体说明

**整体方案**：（1）采用[ESICUP](https://www.euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507)数据集进行排样（2）对给定的图像根据重心最低原则处理，然后通过算法对其形状及大小用向量进行评估（3）通过启发式算法排样，选择最优的多个排样结果输出，包括重心位置/配件方向及利用率（4）对初始输入的参数和输出的结果进行学习，建立映射关系（配件的方向和位置）（5）对初始解进行从下到上的优化，获得最优解

**细节调整**：（1）启发式算法现在只适应水平底部和给定边框（2）配件需要用重心策略进行方向调整后才能进行评估（3）欧式距离小于一定的值可以判定为配件类似

**重点算法**：（1）启发式排样策略（2）配件方向初始化（3）配件的评估参数（4）学习网络怎么实现（没有找到对应网络）

**传统组合优化+机器学习**：（1）阿里巴巴排箱算法，采用Pointer Network（Seq2Seq+Attention），但是只预测添加顺序，对具体排样结果不做预测，可以说是将二维转为一维，也可以说是没办法只能这么做了（2）Georgia Tech的提出的Embedding Graph的算法，主要面向是Travle salesmen这种问题，只面向于基于图的算法 

**算法优势**：（1）可以通过内容直接预测最终的大致位置与方向，以及整体排料效果（2）可以大大提高排样的速度（3）可能可以提高排样效果（4）可以为大规模排样分解做铺垫工作

**难点**：很多难点......

**解决方案**：一步步来

**部分备注**：（1）先从小规模训练起（两个）（2）可以提前采用监督学习，再采用强化学习，神经网络是BP网络



### 简述思路

- [ ] 二维问题过于复杂，部分pairing的方案提出



### 学习网络-10.25

- 输入必须是扁平化的输入，需要很多个神经元
- 首先只考虑两个样片的排序
- 只要你能够解决超小规模的matching就是胜利！能做几个都行！



### 论文

- [ ] Introduction

  - [ ] 2D ISP问题是什么问题，如何解决，有什么应用
  - [ ] 现阶段有哪些解决方案，一般是怎么解决的，遇到了什么问题（寻找算法麻烦，计算量大时间长）
  - [ ] 一些方案因此提出，比如2013的那篇，通过布谷鸟算法来解决，但是仅限于两个pieces
  - [ ] 由此受到启发，如果能够应用到更多的配件上就可以
- [ ] Literature Review 暂时不写

  - [ ] 2D Irregular Strips Packing
  - [ ] Neural Network in Combination Optimization
- [ ] The Difinition of the Problem 暂时不写
  - [ ] 需要具体研究一下再去写——有点麻烦
- [ ] Algorithm Outline
  - [ ] 该节写算法内容
  - [ ] 第一个部分是对样片进行分类的算法
  - [ ] 第二个部分是通过BP以及数据集进行学习的算法——启发式排序算法放在后面，优化算法暂时不考虑
- [ ] Initialization and Evaluation Model
  - [ ] Introduction
  - [ ] 这个部分主要说明如何说明和评价
    - [ ] 主要先通过x x x
  - [ ] Ajdust strips using lowest-gravity-center principle 
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



### 刊物记录

Journal of Zhejiang University-SCIENCE：2016年来看审稿比较快，平均3个月，发表另计，SCI 3区，影响因子1.879，官网www.springer.com/biomed/journal/11585，投稿地址www.editorialmanager.com/zusb/，那篇二维排料论文就是发表在这里

COMPUTERS & OPERATIONS RESEARCH：SCI 3区，计算机跨学科应用/工业/运筹，方向完全匹配，应该有类似方向论文，但是据说变成水刊了.....

Journal of Grey System：偏向于数学与应用数学，审稿比较快，4区



## 排料算法研究计划

### 最终目标

给出一个随机的不规则形状100+个，通过算法对其进行聚类，如ABCDEFGHIJK类，并对几个类别的组合最优组合情况进行预测，如AABBCCCD预计可以达到90%的利用率，即可通过列生成算法进行组合设计。

### 分解需求-2019年10月19日

No.1：需要能够对给出的小规模序列快速预测排料效果——实现了的话，可以将二维问题转化为一维的Stock Cutting Problem

No.2：能够实现在宽度限制下的小规模快速排样——为下一步做铺垫

No.3：能够实相对大规模的组合——在1/2步的基础上，该问题将可以用带限制的一维组合优化求解

No.4：大规模排样后的优化——面临的优化困境

总结：第一步还是类似于阿里巴巴的排料研究的



### 2019年10月12日制作-需要更新

**现阶段核心要点**——快速排样

- 直接学习排料结果
- 核心突破点是快速排序
- 面向的是小规模排样、差异性小的



**下一阶段核心要点**——组合目标的多样性

- 核心突破点是多类别排样
- 通过阶段组合效果进行取舍
- 能够保证效果就是胜利



**下下阶段**——有宽度限制的小规模排样

- 核心突破排样效果限制
- 有目标效果的排样
- 对特定的排样效果学习



**下下下阶段**——相对大规模的组合

- 核心突破点是如何进行排样后的组合
- 主要通过列生成算法
- 用一维问题解决
- 目标函数是组合后的结果



**下下下下阶段**——优化算法

- 核心突破是优化算法
- 利用exchange做heuristic算法