##算法总结

### 搜索算法

#### 局部检索（Local Search）



#### 变邻域检索（Variable Neighborhood Search）



#### 迭代邻域检索（Iterated Local Search, ）



### 精确算法

#### 分支定界法（Branch and Bound）



#### 割平面算法（Branch and Cut）



#### 动态规划算法



### 近似算法

#### 爬山法（Hill Climbing）

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

#### 模拟退火（Simulate Annealing）

局部检索算法，



#### 禁忌搜索（Taboo Search）

局部检索算法，





#### 蚁群算法



#### 遗传算法







### 超启发式算法（Hyper Heuristic）





### 大规模分解算法

#### 列生成（Column Generation）



#### Benders Decomposition





#### Dantzig-Wolfe Decomposition





#### 分支定界法（Branch and Price）





### 加强学习（Reinforcement Learning）







### 其他未分类

#### 拉格朗日松弛



#### 贪婪算法 

