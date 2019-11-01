---
layout: post
title: "运筹学算法"
subtitle: 'Operation Research Algorithm'
author: "sean"
header-img: "img/post-bg-algorithm.jpeg"
tags:
  - 算法
  - 运筹学
---



> ✅ ：已经整理好
>
> 💤 ：大部分整理好了但是没理解
>
> 🆘 ：还有一部分需要整理
>
> ☣️ ：需要整理

## 搜索算法

### ☣️局部检索（Local Search）



### ☣️变邻域检索（Variable Neighborhood Search）



### ☣️迭代邻域检索（Iterated Local Search, ）



## 精确算法

### ☣️分支定界法（Branch and Bound）



### ☣️割平面算法（Branch and Cut）



### ☣️动态规划算法



## 排料算法应用

### ☣️ Not-fit Polygon



### ☣️ 碰撞法



### ☣️ 左底部算法



## 大规模分解算法

### ☣️ 列生成（Column Generation）



### ☣️ Benders Decomposition





### ☣️ Dantzig-Wolfe Decomposition





### ☣️ 分支定价法（Branch and Price）



## 经典问题

### ☣️ 旅行商问题

> 采用进化算法解决

```python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

N_CITIES = 20  # 城市数量
CROSS_RATE = 0.1  # DNA交叉概率
MUTATE_RATE = 0.02  # 变异概率
POP_SIZE = 500  # 个体数量
N_GENERATIONS = 500  # 迭代轮数


class GA:
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size
        #  初始化每个个体的DNA
        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):   # 获得每个城市的坐标
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty(self.pop_size, dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness = np.exp(self.DNA_size * 2 / total_distance)
        return fitness, total_distance

    def select(self, fitness):  # 适应度高的DNA有更大的概率保留
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        return self.pop[idx]

    def crossover(self, parent, pop):
        if np.random.rand() < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)  # 选择另一个体
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)  # 确定DNA如何交叉
            keep_city = parent[~cross_points]
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:  # 变异：交换两个基因
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        for parent in pop:  # for every parent
            child = self.crossover(parent, pop_copy)
            child = self.mutate(child)
            parent[:] = child
        self.pop = pop


class TravelSalesPerson:
    def __init__(self, n_cities):
        self.city_position = np.random.rand(n_cities, 2)  # 初始化每个城市的坐标
        plt.ion()

    def plotting(self, lx, ly, total_d):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.pause(0.01)

ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
env = TravelSalesPerson(N_CITIES)  # 初始化每个城市的坐标

for generation in range(N_GENERATIONS):
    lx, ly = ga.translateDNA(ga.pop, env.city_position)  # 将DNA中的城市id转换为城市坐标
    fitness, total_distance = ga.get_fitness(lx, ly)
    ga.evolve(fitness)  # 模拟进化过程
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx],)

    env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx])

plt.ioff()
plt.show()
```

### ☣️ 0-1 背包问题

> 进化算法解决

```python
# -*- coding: utf-8 -*-
import numpy as np

N_ITEMS = 20  # 物品数量
CROSS_RATE = 0.5  # DNA交叉概率
MUTATE_RATE = 0.01  # 变异概率
POP_SIZE = 400  # 个体数量
N_GENERATIONS = 20  # 迭代轮数


class GA:
    def __init__(self, DNA_size, cross_rate, mutate_rate, pop_size, n_generations):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.pop_size = pop_size
        self.n_generations = n_generations
        #  初始化每个个体的DNA
        self.pop = np.random.randint(0, 2, size=(pop_size, DNA_size))

    def get_fitness(self, w, v, c):
        fitness = np.empty(shape=self.pop_size, dtype=np.float32)
        for i, individual in enumerate(self.pop):  # 计算每个个体的fitness和weight
            weight = w[individual.astype(np.bool)].sum()
            value = v[individual.astype(np.bool)].sum()
            # 若物品总重量超过capacity，则fitness为0
            fitness[i] = 0 if weight > c else value
        return np.exp(fitness/self.DNA_size)

    def select(self, fitness):  # 适应度高的DNA有更大的概率保留
        index = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness/fitness.sum())
        return self.pop[index]

    def mutate(self, dna):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                dna[point] = 0 if dna[point] == 0 else 1  # 变异

    def crossover(self, pop):
        pop_copy = pop.copy()
        for i in range(self.pop_size):
            if np.random.rand() < self.cross_rate:
                i_ = np.random.randint(self.pop_size, size=1)  # 选择另一个体
                cross_points = np.random.randint(2, size=self.DNA_size).astype(np.bool)  # 确定DNA交叉点
                pop[i, cross_points] = pop_copy[i_][0][cross_points]
            self.mutate(pop[i])


c = 40  # 背包容量
w = np.array([4, 5, 6, 2, 2, 3, 7, 8, 6, 9, 1, 5, 6, 8, 3, 4, 4, 9, 4, 2])  # 物品重量
v = np.array([6, 4, 5, 3, 6, 2, 4, 4, 9, 5, 3, 7, 8, 3, 6, 8, 4, 2, 3, 9])  # 物品价值

ga = GA(N_ITEMS, CROSS_RATE, MUTATE_RATE, POP_SIZE, N_GENERATIONS)

for generation in range(N_GENERATIONS):
    fitness = ga.get_fitness(w, v, c)
    pop = ga.select(fitness)
    ga.crossover(pop)
    best_idx = np.argmax(fitness)
    print('Gen:', generation, '| best fit: %.2f' % (np.log(fitness[best_idx])*ga.DNA_size), '| items:', ga.pop[best_idx])
    ga.pop = pop
```

## 其他未分类

### ✅ 启发式算法

![Heuristic-Algorithms](https://tva1.sinaimg.cn/large/006y8mN6ly1g83p9zif17j30iy0g676d.jpg)

### ☣️ 拉格朗日松弛



### ☣️ 贪婪算法 



### ☣️ 网络流算法



### ☣️ 并行计算

