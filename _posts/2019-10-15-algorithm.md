---
layout: post
title: "排料算法整理"
subtitle: 'Slove 2D bin packing'
author: "sean"
header-img: "img/post-bg-unix-linux.jpg"
tags:
  - 算法

---



- [ ] 传统排料算法怎么做的：
- [ ] 阿里的输入与输出具体什么情况：通过传统算法进行排列，并选择最好的几个结果进行评估，然后可以训练出一个序列学习的模型，可以给出最优的加入策略
- [ ] 阿里算法流程：输入箱子的参数，直接给出一个排序序列，按照这个序列把箱子给放进去，通过启发式算法（贪婪策略）进行箱子的排列
- [ ] **算法折中策略**：（1）对于输出的序列他们采用的是对输出的序列进行排序 （2）即使是几个箱子，算法采用的也是贪婪算法，更小的局部最优 （3）本质上对位置没有预测
- [ ] 数据集的问题：http://euro-online.org/websites/esicup/data-sets/#1535972088237-bbcb74e3-b507



确定方案：（1）通过已有的数据集，进行排料，然后选择最优的几个输出策略，包括组合形状和利用率（2）对新给定的排样内容进行评估，

技术优势：（1）可以通过内容直接预测最终的大致位置与方向，以及进行排料（2）







### Comparision

| **Article**          | **Key Note**                                                 | **Thought/Comments**                                         |
| -------------------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
| Bin Packing with DRL | **Input**: a sequence of size data (length, width and height) of items to be packed **Output**: another sequence which represents the order we pack those items **Evaluation**: We use the surface area (SA) of the bin to evaluate the sequence, and we use SA(o\|s) to denote the surface area | This algorithm use aim dataset to train the network, so it needs a  lot time to train the RNN and doesn't fit new shapes combination |

