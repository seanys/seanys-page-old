---
layout: post
title: "排料算法整理和设计"
subtitle: 'Slove 2D bin packing'
author: "sean"
header-img: "img/post-bg-unix-linux.jpg"
tags:
  - 算法

---



## 11月11日投稿安排

论文初稿：11月7日

论文修改：11月8日

论文二改：11月9日

主要目标：样片的匹配，先做最规则的匹配，再去做相对复杂的样片匹配，不要考虑用CNN提取特征，一步步来，或者单纯就做pairing！！！下一步再做放在底部的样片匹配，然后是底部bounded的匹配



## 算法落实

### 量化算法

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g8essqa7faj30u016740l.jpg" alt="image-20191029090930252" style="zoom:50%;" />

**数据处理**：基于多边形的边界点，处理成成边界为2，内部为1的二维矩阵

**一维嵌入**：将二维的矩阵通过区域关系，将其嵌入一个一维矩阵（正在研究）

**参考**：计算几何、Graph/Word Embedding

### 效果衡量

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g8et3fxpcsj30y20hy77u.jpg" alt="image-20191029091949521" style="zoom:50%;" />

**单个对象**：最小外包凸多边形

**多个对象**：样片并集的最小外包凸多边形

**参考**：A new approach for sheet nesting problem using guided cuckoo search and pairwise clustering、计算几何

### 数据集

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g8esoivq8gj30sq0soq8a.jpg" alt="image-20191029090528362" style="zoom:50%;" />

**格式**：CVS格式

**参考数据**：输入两个正方形，输出两个正方形

**输入**：两个多边形，最大值为1（参考边界），输出样片排样后的重心位置

**多选择输入**：同一组样片的组合可能会有不同的组合模式，并且有不同的利用率，所以需要将多个数据集及其评价输入

**参考**：Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method、计算几何、Reinforcement Learning

### 网络

![image-20191029090447541](https://tva1.sinaimg.cn/large/006y8mN6ly1g8esnta2f1j31dc0u01hm.jpg)

**网络**：激活函数采用relu函数，交叉熵损失函数，其他的暂时未确定，训练20轮，检测其效果，暂时采用的是最简单的网络模型

**初步拟合效果**：拟合效果还可以，应该是因为数据是简单的线性回归

**进一步**：需要尽快换成更正式的数据，进行2/3/4个样片的组合测试

## 算法概述

![矩形@3x](https://tva1.sinaimg.cn/large/006y8mN6ly1g8etee1u60j30u00zi4qq.jpg)

## Pairwise Clustering

### 算法概述

排样策略：左底部策略

预处理：匹配算法

优化策略：去除重叠、布谷鸟检索

### 匹配算法

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8gf0o2nenj30ps0iwq5m.jpg" alt="image-20191030184345828" style="zoom:50%;" />

![image-20191030184410099](https://tva1.sinaimg.cn/large/006y8mN6gy1g8gf0zmqrsj31cc0gs77u.jpg)

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8gf16hdfxj30pu0pa0wi.jpg" alt="image-20191030184421216" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8gf1dz5ptj30xq0iwgpc.jpg" alt="image-20191030184433056" style="zoom:50%;" />





## NFP+AG

### SVGNester

<img src="https://camo.githubusercontent.com/f7973d894432676e37c3489c3248c3a31cf3e945/687474703a2f2f7376676e6573742e636f6d2f6769746875622f6e6670322e706e67" alt="No Fit Polygon example" style="zoom:50%;" />

In our GA the insertion order and the rotation of the parts form the gene. The fitness function follows these rules:

1. Minimize the number of unplaceable parts (parts that cannot fit any bin due to its rotation)
2. Minimize the number of bins used
3. Minimize the *width* of all placed parts

The third one is rather arbitrary, as we can also optimize for rectangular bounds or a minimal concave hull. In real-world use the material to be cut tends to be rectangular, and those options tend to result in long slivers of un-used material.

Because small mutations in the gene cause potentially large changes in overall fitness, the individuals of the population can be very similar. By caching NFPs new individuals can be evaluated very quickly.

说明：采用的也是GA算法，一个个放入形状，采用NFP算法计算可行位置，放入的顺序和旋转的角度组成了基因，然后采用变异和遗传来进行优化。

### 刘胡瑶的算法

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8g8b4oeklj316i0pi1fa.jpg" alt="image-20191030145134607" style="zoom:50%;" />

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8g8b9cflvj312o0j0dvx.jpg" alt="image-20191030145151285" style="zoom:50%;" />



## 10月初研究

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



## 论文

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



## 刊物记录

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