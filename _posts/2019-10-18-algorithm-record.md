---
layout: post
title: "算法大杂烩"
subtitle: 'I do not know what I have done'
author: "sean"
header-img: "img/post-bg-os-metro.jpg"
tags:
  - 算法
---



## 搜索算法

### ㊙️局部检索（Local Search）



### ㊙️变邻域检索（Variable Neighborhood Search）



### ㊙️迭代邻域检索（Iterated Local Search, ）



## 精确算法

### ㊙️分支定界法（Branch and Bound）



### ㊙️割平面算法（Branch and Cut）



### ㊙️动态规划算法



## 近似算法

### ㊙️爬山法（Hill Climbing）

Hill climbing finds optimal solutions for convex problems – for other problems it will find only local optima (solutions that cannot be improved upon by any neighboring configurations), which are not necessarily the best possible solution (the global optimum) out of all possible solutions (the search space). 

```scheme
Continuous Space Hill Climbing Algorithm
   currentPoint = initialPoint;    // the zero-magnitude vector is common
   stepSize = initialStepSizes;    // a vector of all 1's is common
   acceleration = someAcceleration; // a value such as 1.2 is common
   candidate[0] = -acceleration;
   candidate[1] = -1 / acceleration;
   candidate[2] = 0;
   candidate[3] = 1 / acceleration;
   candidate[4] = acceleration;
   loop do
      before = EVAL(currentPoint);
      for each element i in currentPoint do
         best = -1;
         bestScore = -INF;
         for j from 0 to 4         // try each of 5 candidate locations
            currentPoint[i] = currentPoint[i] + stepSize[i] * candidate[j];
            temp = EVAL(currentPoint);
            currentPoint[i] = currentPoint[i] - stepSize[i] * candidate[j];
            if(temp > bestScore)
                 bestScore = temp;
                 best = j;
         if candidate[best] is 0
            stepSize[i] = stepSize[i] / acceleration;
         else
            currentPoint[i] = currentPoint[i] + stepSize[i] * candidate[best];
            stepSize[i] = stepSize[i] * candidate[best]; // accelerate
      if (EVAL(currentPoint) - before) < epsilon 
         return currentPoint;
```

### ㊙️模拟退火（Simulate Annealing）

局部检索算法，



### ㊙️禁忌搜索（Taboo Search）

局部检索算法，





### ㊙️蚁群算法



### ㊙️遗传算法







## 超启发式算法（Hyper Heuristic）





## 大规模分解算法

### ㊙️列生成（Column Generation）



### ㊙️Benders Decomposition





### ㊙️Dantzig-Wolfe Decomposition





### ㊙️分支定界法（Branch and Price）



## 排料算法应用

### ㊙️Not-fit Polygon



### ㊙️碰撞法



### ㊙️左底部算法





## 加强学习（Reinforcement Learning）



## 机器学习

### ㊙️注意力机制（Attention Mechanism）





## 统计与回归模型

### ㊙️线性规划（Linear Regression）



### ㊙️逻辑回归（Logistic Regression）

![img](https://yuanxiaosc.github.io/2018/06/21/%E6%94%B9%E8%BF%9B%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E2%80%94%E2%80%94%E4%BA%A4%E5%8F%89%E7%86%B5/one_1.png)



### ㊙️SoftMax

![“softmax”的图片搜索结果](https://pic1.zhimg.com/v2-11758fbc2fc5bbbc60106926625b3a4f_1200x500.jpg)

```python
scores = np.array([123, 456, 789])    # example with 3 classes and each having large scores
scores -= np.max(scores)    # scores becomes [-666, -333, 0]
p = np.exp(scores) / np.sum(np.exp(scores))
```

主要用于多分类回归，可以对向量进行处理，最后得出的概率总和是1

### ㊙️似然函数（Likelihood Function）





参考资料：https://cloud.tencent.com/developer/article/1102103（分类算法总结）、https://zhuanlan.zhihu.com/p/25723112（softmax应用）

## 激活函数

### ㊙️Relu Function

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ramp_function.svg/325px-Ramp_function.svg.png)



## 损失函数

### ㊙️交叉熵损失函数





## 其他未分类

### ㊙️拉格朗日松弛



### ㊙️贪婪算法 

