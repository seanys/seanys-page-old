---
layout: post
title: "基于Ptr-Network的二维不规则形状快速排料算法-设计中"
subtitle: 'Fast Algorithm for 2D Irregular Shape Packing based on Ptr-Network'
author: "sean"
header-img: "img/post-bg-ai.jpeg"
tags:
  - 算法
  - 机器学习
  - 组合优化
---



### Initial Model

$$Status_i=\begin{vmatrix}Postion \\\\ Size \\\\ Orientation \\\\ Feature_1 \\\\ Feature_2 \\\\ Feature_3 \\\\ \vdots \end{vmatrix}$$

 We can give every items a matrix which can show its status. 



### Dataset

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g83i4bwxv2j31mu0u00xq.jpg" style="zoom:30%"/>

### Learning Model 

![model](https://tva1.sinaimg.cn/large/006y8mN6ly1g83k7vcqwgj31id0u0gwg.jpg)

Plan proposed by Alibaba which can solve  3D bin packing problem is based on 