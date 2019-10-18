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

### ☣️局部检索（Local Search）



### ☣️变邻域检索（Variable Neighborhood Search）



### ☣️迭代邻域检索（Iterated Local Search, ）



## 精确算法

### ☣️分支定界法（Branch and Bound）



### ☣️割平面算法（Branch and Cut）



### ☣️动态规划算法



## 近似算法

### 前言

**问题一**：航班选择最优，共计12个航班，要使得总的票价最少且每个人的等待时间之和最小

**问题二**：学生选择宿舍的问题，每个学生可以实现填报志愿，如果安排的宿舍与志愿完全一致，则代价为0，与第二志愿一致，代价为1，如果没有和志愿一致，代价为3。 

**核心**：将自己要优化的目标量化的表达出来，是解决优化函数的关键

**损失函数**：如在普通的数值优化问题中，可选择当前值与目标值距离的绝对之差，或者使用平方损失函数均可。

代码参考资料：https://blog.csdn.net/bcj296050240/article/details/50839806

### ✅随机搜索算法

```python
# 搜索方法1: 随机搜索算法
# 函数会作1000次随机猜测，记录总代价最低的方法. domain为航班号的范围（0-9），共有5个人，因此共有10项
def randomoptimize(self, domain):
    best_sol = []
    bestcost = 99999
    for i in range(1000):
        sol = [0] * len(domain)
        for j in range(len(domain) / 2):  # 人数
            sol[2 * j] = random.randint(domain[2 * j][0], domain[2 * j][1])
            sol[2 * j + 1] = random.randint(domain[2 * j + 1][0], domain[2 * j + 1][1])

        print sol[:]
        newcost = self.schedulecost(sol)
        if newcost < bestcost:
            bestcost = newcost
            best_sol = sol
        else:
            continue
    self.printschedule(best_sol)
    print "随机搜索算法的结果的最小代价是：", bestcost
    return best_sol
```

总结：重点在随机函数，相当于大量枚举但是不连续

### ✅爬山法（Hill Climbing）

```python
    # 搜索算法2：爬山法
    # 首先随机选择一个解作为种子解，每次寻找这个种子相近的解，如果相近的解有代价更小的解，则把这个新的解作为种子
    # 依次循环进行，当循环到某种子附近的解都比该种子的代价大时，说明到达了局部极小值点，搜索结束
    def hillclimb(self, domain):
        # 随机产生一个航班序列作为初始种子
        seed = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        while 1:
            neighbor = []
            # 循环改变解的每一个值产生一个临近解的列表
            for i in range(len(domain)):
                # 下列判断是为了将某一位加减1后不超出domain的范围
                # print seed
                if seed[i] > domain[i][0]:
                    newneighbor = seed[0:i] + [seed[i] - 1] + seed[i + 1:]
                    # print newneighbor[:]
                    neighbor.append(newneighbor)
                if seed[i] < domain[i][1]:
                    newneighbor = seed[0:i] + [seed[i] + 1] + seed[i + 1:]
                    # print newneighbor[:]
                    neighbor.append(newneighbor)

            # 对所有的临近解计算代价，排序，得到代价最小的解
            neighbor_cost = sorted(
                [(s, self.schedulecost(s)) for s in neighbor], key=lambda x: x[1])

            # 如果新的最小代价 > 原种子代价，则跳出循环
            if neighbor_cost[0][1] > self.schedulecost(seed):
                break

            # 新的代价更小的临近解作为新的种子
            seed = neighbor_cost[0][0]
            print "newseed = ", seed[:], " 代价：", self.schedulecost(seed)
        # 输出
        self.printschedule(seed)
        print "爬山法得到的解的最小代价是", self.schedulecost(seed)
        return seed
```

**总结**：对于单峰的优化问题是可以的，对于多峰问题可能会陷入局部最优化，因为只会检索临近点并计算效果。其次，爬山的问题在于如何实现“临近”，比如排料中的NFP就可以模拟“临近”的效果。

### 🆘模拟退火（Simulate Annealing）

```python
    # 搜索算法4：模拟退火算法
    # 参数：T代表原始温度，cool代表冷却率，step代表每次选择临近解的变化范围
    # 原理：退火算法以一个问题的随机解开始，用一个变量表示温度，这一温度开始时非常高，而后逐步降低
    #      在每一次迭代期间，算法会随机选中题解中的某个数字，然后朝某个方向变化。如果新的成本值更
    #      低，则新的题解将会变成当前题解，这与爬山法类似。不过，如果成本值更高的话，则新的题解仍
    #      有可能成为当前题解，这是避免局部极小值问题的一种尝试。
    # 注意：算法总会接受一个更优的解，而且在退火的开始阶段会接受较差的解，随着退火的不断进行，算法
    #      原来越不能接受较差的解，直到最后，它只能接受更优的解。
    # 算法接受较差解的概率 P = exp[-(highcost-lowcost)/temperature]
    def annealingoptimize(self, domain, T=10000.0, cool=0.98, step=1):
        # 随机初始化值
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

        # 循环
        while T > 0.1:
            # 选择一个索引值
            i = random.randint(0, len(domain) - 1)
            # 选择一个改变索引值的方向
            c = random.randint(-step, step)  # -1 or 0 or 1
            # 构造新的解
            vecb = vec[:]
            vecb[i] += c
            if vecb[i] < domain[i][0]:  # 判断越界情况
                vecb[i] = domain[i][0]
            if vecb[i] > domain[i][1]:
                vecb[i] = domain[i][1]

            # 计算当前成本和新的成本
            cost1 = self.schedulecost(vec)
            cost2 = self.schedulecost(vecb)

            # 判断新的解是否优于原始解 或者 算法将以一定概率接受较差的解
            if cost2 < cost1 or random.random() < math.exp(-(cost2 - cost1) / T):
                vec = vecb

            T = T * cool  # 温度冷却
            print vecb[:], "代价:", self.schedulecost(vecb)

        self.printschedule(vec)
        print "模拟退火算法得到的最小代价是：", self.schedulecost(vec)
        return vec
```

总结：

### 🆘遗传算法

```python
# 搜索算法5： 遗传算法
# 原理： 首先随机生成一组解，我们称之为种群，在优化过程的每一步，算法会计算整个种群的成本函数，
#       从而得到一个有关题解的有序列表。随后根据种群构造进化的下一代种群，方法如下：
#       遗传：从当前种群中选出代价最优的一部分加入下一代种群，称为“精英选拔”
#       变异：对一个既有解进行微小的、简单的、随机的修改
#       交叉：选取最优解中的两个解，按照某种方式进行交叉。方法有单点交叉，多点交叉和均匀交叉
# 一个种群是通过对最优解进行随机的变异和配对处理构造出来的，它的大小通常与旧的种群相同，尔后，这一过程会
#       一直重复进行————新的种群经过排序，又一个种群被构造出来，如果达到指定的迭代次数之后题解都没有得
#       改善，整个过程就结束了
# 参数：
# popsize-种群数量 step-变异改变的大小 mutprob-交叉和变异的比例 elite-直接遗传的比例 maxiter-最大迭代次数
def geneticoptimize(self, domain, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    # 变异操作的函数
    def mutate(vec):
        i = random.randint(0, len(domain) - 1)
        if random.random() < 0.5 and vec[i] > domain[i][0]:
            return vec[0:i] + [vec[i] - step] + vec[i + 1:]
        elif vec[i] < domain[i][1]:
            return vec[0:i] + [vec[i] + step] + vec[i + 1:]

    # 交叉操作的函数（单点交叉）
    def crossover(r1, r2):
        i = random.randint(0, len(domain) - 1)
        return r1[0:i] + r2[i:]

    # 构造初始种群
    pop = []
    for i in range(popsize):
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        pop.append(vec)

    # 每一代中有多少胜出者
    topelite = int(elite * popsize)

    # 主循环
    for i in range(maxiter):
        scores = [(self.schedulecost(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s, v) in scores]  # 解按照代价由小到大的排序

        # 优质解遗传到下一代
        pop = ranked[0: topelite]

        # 如果当前种群数量小于既定数量，则添加变异和交叉遗传
        while len(pop) < popsize:
            # 随机数小于 mutprob 则变异，否则交叉
            if random.random() < mutprob:  # mutprob控制交叉和变异的比例
                # 选择一个个体
                c = random.randint(0, topelite)
                # 变异
                pop.append(mutate(ranked[c]))
            else:
                # 随机选择两个个体进行交叉
                c1 = random.randint(0, topelite)
                c2 = random.randint(0, topelite)
                pop.append(crossover(ranked[c1], ranked[c2]))
        # 输出当前种群中代价最小的解
        print scores[0][1], "代价：", scores[0][0]
    self.printschedule(scores[0][1])
    print "遗传算法求得的最小代价：", scores[0][0]
    return scores[0][1]
```

总结：



### ☣️禁忌搜索（Taboo Search）

局部检索算法，





### ☣️蚁群算法



### 🆘综合案例

```python
# -*-  coding: utf-8 -*-
__author__ = 'Bai Chenjia'

import random
import math
"""
宿舍分配问题，属于搜索优化问题。优化方法使用optimization.py中使用的
随机搜索、爬山法、模拟退火法、遗传算法等. 但题解的描述比之前的问题复杂
"""


class dorm:
    def __init__(self):
        # 代表宿舍，每个宿舍有两个隔间可用
        self.dorms = ['Zeus', 'Athena', 'Hercules', 'Bacchus', 'Pluto']

        # 代表选择，第一列代表人名，第二列和第三列代表该学生的首选和次选
        self.prefs = [('Toby', ('Bacchus', 'Hercules')),
                      ('Steve', ('Zeus', 'Pluto')),
                      ('Karen', ('Athena', 'Zeus')),
                      ('Sarah', ('Zeus', 'Pluto')),
                      ('Dave', ('Athena', 'Bacchus')),
                      ('Jeff', ('Hercules', 'Pluto')),
                      ('Fred', ('Pluto', 'Athena')),
                      ('Suzie', ('Bacchus', 'Hercules')),
                      ('Laura', ('Bacchus', 'Hercules')),
                      ('James', ('Hercules', 'Athena'))]

        # 题解的表示法：
        # 设想每个宿舍有两个槽，本例中共有5个宿舍，则共有10个槽，我们将每名学生依序安置于各空槽
        # 内————则第一名学生有10种选择，解的取值范围为0-9；第二名学生有9种选择，解的取值范围为
        # 0-8，第三名学生解的取值范围是0-7，以此类推，最后一名学生只有一个可选。
        # 按照上述题解的表示法初始化题解范围：
        self.domain = [(0, 2*len(self.dorms)-1-i) for i in range(len(self.prefs))]

    # 根据题解序列vec打印出最终宿舍分配方案
    # 注意，输出一个槽后表明该槽已经用过，需将该槽删除
    def printsolution(self, vec):
        slots = []
        # slots = [0,0,1,1,2,2,3,3,4,4]
        for i in range(len(self.dorms)):
            slots.append(i)
            slots.append(i)
        # 循环题解
        print "选择方案是："
        for i in range(len(vec)):
            index = slots[vec[i]]
            print self.prefs[i][0], self.dorms[index]
            del slots[vec[i]]

    # 代价函数: 如果学生当前安置的宿舍使其首选则代价为0，是其次选则代价为1，否则代价为3
    # 注意，输出一个槽后表明该槽已经用过，需将该槽删除
    def dormcost(self, vec):
        cost = 0
        # 建立槽
        slots = []
        for i in range(len(self.dorms)):
            slots.append(i)
            slots.append(i)
        # 循环题解
        for i in range(len(vec)):
            index = slots[vec[i]]
            if self.dorms[index] == self.prefs[i][1][0]:
                cost += 0
            elif self.dorms[index] == self.prefs[i][1][1]:
                cost += 1
            else:
                cost += 3
            del slots[vec[i]]
        return cost

    """
    下列函数与 optimization 中函数相同，只不过代价函数和输出函数用本问题的输出函数
    """
    # 搜索方法1: 随机搜索算法
    # 函数会作1000次随机猜测，记录总代价最低的方法. domain为航班号的范围（0-9），共有5个人，因此共有10项
    def randomoptimize(self, domain):
        best_sol = []
        bestcost = 99999
        for i in range(1000):
            sol = [0] * len(domain)
            for j in range(len(domain) / 2):  # 人数
                sol[2 * j] = random.randint(domain[2 * j][0], domain[2 * j][1])
                sol[2 * j + 1] = random.randint(domain[2 * j + 1][0], domain[2 * j + 1][1])

            print sol[:]
            newcost = self.dormcost(sol)
            if newcost < bestcost:
                bestcost = newcost
                best_sol = sol
            else:
                continue
        self.printsolution(best_sol)
        print "随机搜索算法的结果的最小代价是：", bestcost
        return best_sol

    # 搜索算法2：爬山法
    # 首先随机选择一个解作为种子解，每次寻找这个种子相近的解，如果相近的解有代价更小的解，则把这个新的解作为种子
    # 依次循环进行，当循环到某种子附近的解都比该种子的代价大时，说明到达了局部极小值点，搜索结束
    def hillclimb(self, domain):
        # 随机产生一个航班序列作为初始种子
        seed = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
        while 1:
            neighbor = []
            # 循环改变解的每一个值产生一个临近解的列表
            for i in range(len(domain)):
                # 下列判断是为了将某一位加减1后不超出domain的范围
                # print seed
                if seed[i] > domain[i][0]:
                    newneighbor = seed[0:i] + [seed[i] - 1] + seed[i + 1:]
                    # print newneighbor[:]
                    neighbor.append(newneighbor)
                if seed[i] < domain[i][1]:
                    newneighbor = seed[0:i] + [seed[i] + 1] + seed[i + 1:]
                    # print newneighbor[:]
                    neighbor.append(newneighbor)

            # 对所有的临近解计算代价，排序，得到代价最小的解
            neighbor_cost = sorted(
                [(s, self.dormcost(s)) for s in neighbor], key=lambda x: x[1])

            # 如果新的最小代价 > 原种子代价，则跳出循环
            if neighbor_cost[0][1] > self.dormcost(seed):
                break

            # 新的代价更小的临近解作为新的种子
            seed = neighbor_cost[0][0]
            print "newseed = ", seed[:], " 代价：", self.dormcost(seed)
        # 输出
        self.printsolution(seed)
        print "爬山法得到的解的最小代价是", self.dormcost(seed)
        return seed

    # 搜索算法4：模拟退火算法
    # 参数：T代表原始温度，cool代表冷却率，step代表每次选择临近解的变化范围
    # 原理：退火算法以一个问题的随机解开始，用一个变量表示温度，这一温度开始时非常高，而后逐步降低
    #      在每一次迭代期间，算啊会随机选中题解中的某个数字，然后朝某个方向变化。如果新的成本值更
    #      低，则新的题解将会变成当前题解，这与爬山法类似。不过，如果成本值更高的话，则新的题解仍
    #      有可能成为当前题解，这是避免局部极小值问题的一种尝试。
    # 注意：算法总会接受一个更优的解，而且在退火的开始阶段会接受较差的解，随着退火的不断进行，算法
    #      原来越不能接受较差的解，直到最后，它只能接受更优的解。
    # 算法接受较差解的概率 P = exp[-(highcost-lowcost)/temperature]
    def annealingoptimize(self, domain, T=10000.0, cool=0.98, step=1):
        # 随机初始化值
        vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

        # 循环
        while T > 0.1:
            # 选择一个索引值
            i = random.randint(0, len(domain) - 1)
            # 选择一个改变索引值的方向
            c = random.randint(-step, step)  # -1 or 0 or 1
            # 构造新的解
            vecb = vec[:]
            vecb[i] += c
            if vecb[i] < domain[i][0]:  # 判断越界情况
                vecb[i] = domain[i][0]
            if vecb[i] > domain[i][1]:
                vecb[i] = domain[i][1]

            # 计算当前成本和新的成本
            cost1 = self.dormcost(vec)
            cost2 = self.dormcost(vecb)

            # 判断新的解是否优于原始解 或者 算法将以一定概率接受较差的解
            if cost2 < cost1 or random.random() < math.exp(-(cost2 - cost1) / T):
                vec = vecb

            T = T * cool  # 温度冷却
            print vecb[:], "代价:", self.dormcost(vecb)

        self.printsolution(vec)
        print "模拟退火算法得到的最小代价是：", self.dormcost(vec)
        return vec

    # 搜索算法5： 遗传算法
    # 原理： 首先随机生成一组解，我们称之为种群，在优化过程的每一步，算法会计算整个种群的成本函数，
    #       从而得到一个有关题解的有序列表。随后根据种群构造进化的下一代种群，方法如下：
    #       遗传：从当前种群中选出代价最优的一部分加入下一代种群，称为“精英选拔”
    #       变异：对一个既有解进行微小的、简单的、随机的修改
    #       交叉：选取最优解中的两个解，按照某种方式进行交叉。方法有单点交叉，多点交叉和均匀交叉
    # 一个种群是通过对最优解进行随机的变异和配对处理构造出来的，它的大小通常与旧的种群相同，尔后，这一过程会
    #       一直重复进行————新的种群经过排序，又一个种群被构造出来，如果达到指定的迭代次数之后题解都没有得
    #       改善，整个过程就结束了
    # 参数：
    # popsize-种群数量 step-变异改变的大小 mutprob-交叉和变异的比例 elite-直接遗传的比例 maxiter-最大迭代次数
    def geneticoptimize(self, domain, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
        # 变异操作的函数
        def mutate(vec):
            i = random.randint(0, len(domain) - 1)
            res = []
            if random.random() < 0.5 and vec[i] > domain[i][0]:
                res = vec[0:i] + [vec[i] - step] + vec[i + 1:]
            elif vec[i] < domain[i][1]:
                res = vec[0:i] + [vec[i] + step] + vec[i + 1:]
            else:
                res = vec
            return res

        # 交叉操作的函数（单点交叉）
        def crossover(r1, r2):
            i = random.randint(0, len(domain) - 1)
            return r1[0:i] + r2[i:]

        # 构造初始种群
        pop = []
        for i in range(popsize):
            vec = [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]
            pop.append(vec)

        # 每一代中有多少胜出者
        topelite = int(elite * popsize)

        # 主循环
        for i in range(maxiter):
            if [] in pop:
                print "***"
            try:
                scores = [(self.dormcost(v), v) for v in pop]
            except:
                print "pop!!", pop[:]
            scores.sort()
            ranked = [v for (s, v) in scores]  # 解按照代价由小到大的排序

            # 优质解遗传到下一代
            pop = ranked[0: topelite]
            # 如果当前种群数量小于既定数量，则添加变异和交叉遗传
            while len(pop) < popsize:
                # 随机数小于 mutprob 则变异，否则交叉
                if random.random() < mutprob:  # mutprob控制交叉和变异的比例
                    # 选择一个个体
                    c = random.randint(0, topelite)
                    # 变异
                    if len(ranked[c]) == len(self.prefs):
                        temp = mutate(ranked[c])
                        if temp == []:
                            print "******", ranked[c]
                        else:
                            pop.append(temp)

                else:
                    # 随机选择两个个体进行交叉
                    c1 = random.randint(0, topelite)
                    c2 = random.randint(0, topelite)
                    pop.append(crossover(ranked[c1], ranked[c2]))
            # 输出当前种群中代价最小的解
            print scores[0][1], "代价：", scores[0][0]
        self.printsolution(scores[0][1])
        print "遗传算法求得的最小代价：", scores[0][0]
        return scores[0][1]

if __name__ == '__main__':
    dormsol = dorm()
    #sol = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #dormsol.printsolution(sol)
    #dormsol.dormcost(sol)

    domain = dormsol.domain

    # 方法1：随机猜测
    #dormsol.randomoptimize(domain)

    # 方法2：爬山法
    #dormsol.hillclimb(domain)

    # 方法3：模拟退火法
    #dormsol.annealingoptimize(domain)

    # 方法4：遗传算法
    dormsol.geneticoptimize(domain)
```

参考资料：https://blog.csdn.net/bcj296050240/article/details/50839806

## 超启发式算法（Hyper Heuristic）





## 大规模分解算法

### ☣️列生成（Column Generation）



### ☣️Benders Decomposition





### ☣️Dantzig-Wolfe Decomposition





### ☣️分支定价法（Branch and Price）



## 排料算法应用

### ☣️Not-fit Polygon



### ☣️碰撞法



### ☣️左底部算法





## 强化学习（Reinforcement Learning）

### ☣️介绍



### ☣️Q-Learning





## 机器学习

### ☣️注意力机制（Attention Mechanism）



##迁移学习





## 统计与回归模型

### ☣️线性规划（Linear Regression）



### 🆘逻辑回归（Logistic Regression）

![img](https://yuanxiaosc.github.io/2018/06/21/%E6%94%B9%E8%BF%9B%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E2%80%94%E2%80%94%E4%BA%A4%E5%8F%89%E7%86%B5/one_1.png)



### 🆘SoftMax

![“softmax”的图片搜索结果](https://pic1.zhimg.com/v2-11758fbc2fc5bbbc60106926625b3a4f_1200x500.jpg)

```python
scores = np.array([123, 456, 789])    # example with 3 classes and each having large scores
scores -= np.max(scores)    # scores becomes [-666, -333, 0]
p = np.exp(scores) / np.sum(np.exp(scores))
```

主要用于多分类回归，可以对向量进行处理，最后得出的概率总和是1

参考资料：https://zhuanlan.zhihu.com/p/25723112（softmax应用）

### ☣️似然函数（Likelihood Function）



### ☣️贝叶斯模型（Bayes）



### ✅马尔可夫模型（Markov Models）

<img src="https://pic4.zhimg.com/80/648a55725e67d718d97d6a475891d70b_hd.jpg" style="width:50%"/>

概述：马尔可夫过程中，任何一个状态，只与前一个状态相关

```python
# 关系矩阵
transfer_matrix = np.array([[0.6,0.2,0.2],[0.3,0.4,0.3],[0,0.3,0.7]],dtype='float32') 
# 
start_matrix = np.array([[0.5,0.3,0.2]],dtype='float32') 

value1 = []
value2 = []
value3 = []
for i in range(30):
    start_matrix = np.dot(start_matrix,transfer_matrix)
    value1.append(start_matrix[0][0])
    value2.append(start_matrix[0][1])
    value3.append(start_matrix[0][2])
print(start_matrix)

# 输出[[0.23076934 0.30769244 0.4615386 ]]
```

### ✅隐马尔可夫模型（Hidden Markov Models）

#### 前言

隐马尔可夫模型即马尔可夫链上加了一层随机过程，一般采用Baum-Welch算法和Viterbi算法进行求解

![preview](https://pic2.zhimg.com/792e033ff9b0418b3b6c9bbaef30fd83_r.jpg)

#### Baum-Welch算法与Viterbi算法



#### 源码-EM算法

```python
"""
隐马尔科夫模型
三类问题：1.概率计算 2.学习问题（参数估计） 3.预测问题（状态序列的预测）
"""
import numpy as np
from itertools import accumulate

class GenData:
    """
    根据隐马尔科夫模型生成相应的观测数据
    """
    def __init__(self, hmm, n_sample):
        self.hmm = hmm
        self.n_sample = n_sample

    def _locate(self, prob_arr):
        # 给定概率向量，返回状态
        seed = np.random.rand(1)
        for state, cdf in enumerate(accumulate(prob_arr)):
            if seed <= cdf:
                return state
        return

    def init_state(self):
        # 根据初始状态概率向量，生成初始状态
        return self._locate(self.hmm.S)

    def state_trans(self, current_state):
        # 转移状态
        return self._locate(self.hmm.A[current_state])

    def gen_obs(self, current_state):
        # 生成观测
        return self._locate(self.hmm.B[current_state])

    def gen_data(self):
        # 根据模型产生观测数据
        current_state = self.init_state()
        start_obs = self.gen_obs(current_state)
        state = [current_state]
        obs = [start_obs]
        n = 0
        while n < self.n_sample - 1:
            n += 1
            current_state = self.state_trans(current_state)
            state.append(current_state)
            obs.append(self.gen_obs(current_state))
        return state, obs


class HMM:
    def __init__(self, n_state, n_obs, S=None, A=None, B=None):
        self.n_state = n_state  # 状态的个数n
        self.n_obs = n_obs  # 观测的种类数m
        self.S = S  # 1*n, 初始状态概率向量
        self.A = A  # n*n, 状态转移概率矩阵
        self.B = B  # n*m, 观测生成概率矩阵

def _alpha(hmm, obs, t):
    # 计算时刻t各个状态的前向概率
    b = hmm.B[:, obs[0]]
    alpha = np.array([hmm.S * b])  # n*1
    for i in range(1, t + 1):
        alpha = (alpha @ hmm.A) * np.array([hmm.B[:, obs[i]]])
    return alpha[0]

def forward_prob(hmm, obs):
    # 前向算法计算最终生成观测序列的概率, 即各个状态下概率之和
    alpha = _alpha(hmm, obs, len(obs) - 1)
    return np.sum(alpha)

def _beta(hmm, obs, t):
    # 计算时刻t各个状态的后向概率
    beta = np.ones(hmm.n_state)
    for i in reversed(range(t + 1, len(obs))):
        beta = np.sum(hmm.A * hmm.B[:, obs[i]] * beta, axis=1)
    return beta

def backward_prob(hmm, obs):
    # 后向算法计算生成观测序列的概率
    beta = _beta(hmm, obs, 0)
    return np.sum(hmm.S * hmm.B[:, obs[0]] * beta)

def fb_prob(hmm, obs, t=None):
    # 将前向和后向合并
    if t is None:
        t = 0
    res = _alpha(hmm, obs, t) * _beta(hmm, obs, t)
    return res.sum()

def _gamma(hmm, obs, t):
    # 计算时刻t处于各个状态的概率
    alpha = _alpha(hmm, obs, t)
    beta = _beta(hmm, obs, t)
    prob = alpha * beta
    return prob / prob.sum()


def point_prob(hmm, obs, t, i):
    # 计算时刻t处于状态i的概率
    prob = _gamma(hmm, obs, t)
    return prob[i]

def _xi(hmm, obs, t):
    alpha = np.mat(_alpha(hmm, obs, t))
    beta_p = _beta(hmm, obs, t + 1)
    obs_prob = hmm.B[:, obs[t + 1]]
    obs_beta = np.mat(obs_prob * beta_p)
    alpha_obs_beta = np.asarray(alpha.T * obs_beta)
    xi = alpha_obs_beta * hmm.A
    return xi / xi.sum()


def fit(hmm, obs_data, maxstep=100):
    # 利用Baum-Welch算法学习
    hmm.A = np.ones((hmm.n_state, hmm.n_state)) / hmm.n_state
    hmm.B = np.ones((hmm.n_state, hmm.n_obs)) / hmm.n_obs
    hmm.S = np.random.sample(hmm.n_state)  # 初始状态概率矩阵（向量），的初始化必须随机状态，否则容易陷入局部最优
    hmm.S = hmm.S / hmm.S.sum()
    step = 0
    while step < maxstep:
        xi = np.zeros_like(hmm.A)
        gamma = np.zeros_like(hmm.S)
        B = np.zeros_like(hmm.B)
        S = _gamma(hmm, obs_data, 0)
        for t in range(len(obs_data) - 1):
            tmp_gamma = _gamma(hmm, obs_data, t)
            gamma += tmp_gamma
            xi += _xi(hmm, obs_data, t)
            B[:, obs_data[t]] += tmp_gamma

        # 更新 A
        for i in range(hmm.n_state):
            hmm.A[i] = xi[i] / gamma[i]
        # 更新 B
        tmp_gamma_end = _gamma(hmm, obs_data, len(obs_data) - 1)
        gamma += tmp_gamma_end
        B[:, obs_data[-1]] += tmp_gamma_end
        for i in range(hmm.n_state):
            hmm.B[i] = B[i] / gamma[i]
        # 更新 S
        hmm.S = S
        step += 1
    return hmm

def predict(hmm, obs):
    # 采用Viterbi算法预测状态序列
    N = len(obs)
    nodes_graph = np.zeros((hmm.n_state, N), dtype=int)  # 存储时刻t且状态为i时， 前一个时刻t-1的状态，用于构建最终的状态序列
    delta = hmm.S * hmm.B[:, obs[0]]  # 存储到t时刻，且此刻状态为i的最大概率
    nodes_graph[:, 0] = range(hmm.n_state)

    for t in range(1, N):
        new_delta = []
        for i in range(hmm.n_state):
            temp = [hmm.A[j, i] * d for j, d in enumerate(delta)]  # 当前状态为i时， 选取最优的前一时刻状态
            max_d = max(temp)
            new_delta.append(max_d * hmm.B[i, obs[t]])
            nodes_graph[i, t] = temp.index(max_d)
        delta = new_delta

    current_state = np.argmax(nodes_graph[:, -1])
    path = []
    t = N
    while t > 0:
        path.append(current_state)
        current_state = nodes_graph[current_state, t - 1]
        t -= 1
    return list(reversed(path))


if __name__ == '__main__':
    # S = np.array([0.2, 0.4, 0.4])
    # A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
    # B = np.array([[0.5, 0.2, 0.3], [0.4, 0.2, 0.4], [0.6, 0.3, 0.1]])
    # hmm_real = HMM(3, 3, S, A, B)
    # g = GenData(hmm_real, 500)
    # state, obs = g.gen_data()
    # 检测生成的数据
    # state, obs = np.array(state), np.array(obs)
    # ind = np.where(state==2)[0]
    # from collections import Counter
    # obs_ind = obs[ind]
    # c1 = Counter(obs_ind)
    # n = sum(c1.values())
    # for o, val in c.items():
    #     print(o, val/n)
    # ind_next = ind + 1
    # ind_out = np.where(ind_next==1000)
    # ind_next = np.delete(ind_next, ind_out)
    # state_next = state[ind_next]
    # c2 = Counter(state_next)
    # n = sum(c2.values())
    # for s, val in c2.items():
    #     print(s, val/n)

    # 预测
    S = np.array([0.5, 0.5])
    A = np.array([[0.8, 1], [0.8, 0.8]])
    B = np.array([[0.2, 0.0, 0.8], [0, 0.8, 0.2]])
    hmm = HMM(2, 3, S, A, B)
    g = GenData(hmm, 200)
    state, obs = g.gen_data()
    print(obs)
    path = predict(hmm, obs)
    score = sum([int(i == j) for i, j in zip(state, path)])
    print(score / len(path))

    # 学习
    # import matplotlib.pyplot as plt
    #
    #
    # def triangle_data(n_sample):
    #     # 生成三角波形状的序列
    #     data = []
    #     for x in range(n_sample):
    #         x = x % 6
    #         if x <= 3:
    #             data.append(x)
    #         else:
    #             data.append(6 - x)
    #     return data
    #
    #
    # hmm = HMM(10, 4)
    # data = triangle_data(30)
    # hmm = fit(hmm, data)
    # g = GenData(hmm, 30)
    # state, obs = g.gen_data()
    #
    # x = [i for i in range(30)]
    # plt.scatter(x, obs, marker='*', color='r')
    # plt.plot(x, data, color='g')
    # plt.show()
```

参考资料：https://www.zhihu.com/question/20962240（马尔可夫）、https://blog.csdn.net/slx_share/article/details/80237566（隐马尔可夫）、https://blog.csdn.net/slx_share/article/details/80237566（源码案例）

### ✅蒙特卡罗方法（**Monte Carlo** Method）

#### 基础

**概述**通过大量随机样本，去了解一个系统，进而得到所要计算的值。它非常强大和灵活，又相当简单易懂，很容易实现。对于许多问题来说，它往往是最简单的计算方法，有时甚至是唯一可行的方法。

**举例**：在区域内随机产生10000个点，落在圆内的点与圆外的点比例计划，可以推导𝝅的近似值

<img src="http://www.ruanyifeng.com/blogimg/asset/2015/bg2015072604.jpg" style="width:30%"/>

#### 拒绝接受采样



#### 源码

```python
import random
def calpai():
    n = 1000000
    r = 1.0
    a, b = (0.0, 0.0)
    x_neg, x_pos = a - r, a + r
    y_neg, y_pos = b - r, b + r

    count = 0
    for i in range(0, n):
        x = random.uniform(x_neg, x_pos)
        y = random.uniform(y_neg, y_pos)
        if x*x + y*y <= 1.0:
            count += 1

    print (count / float(n)) * 4
```

参考资料：http://www.ruanyifeng.com/blog/2015/07/monte-carlo-method.html 蒙特卡洛

### 🆘马尔可夫链蒙特卡洛（Markov chain Monte Carlo）

#### 背景

**动机一**

假如你需要对一维随机变量$X$进行采样， ![[公式]](https://www.zhihu.com/equation?tex=X) 的样本空间是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2C2%2C3%5C%7D) ，且概率分别是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2F2%2C1%2F4%2C1%2F4%5C%7D) ，这很简单，只需写这样简单的程序：首先根据各离散取值的概率大小对 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D)区间进行等比例划分，如划分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C0.5%5D%2C%5B0%2C5%2C0.75%5D%2C%5B0.75%2C1%5D) 这三个区间，再通过计算机产生 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 之间的伪随机数，根据伪随机数的落点即可完成一次采样。接下来，假如 ![[公式]](https://www.zhihu.com/equation?tex=X) 是连续分布的呢，概率密度是 ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) ，那该如何进行采样呢？聪明的你肯定会想到累积分布函数， ![[公式]](https://www.zhihu.com/equation?tex=P%28X%3Ct%29%3D%5Cint+_%7B-%5Cinfty%7D%5E%7Bt%7Df%28x%29dx) ，即在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 间随机生成一个数 ![[公式]](https://www.zhihu.com/equation?tex=a) ，然后求使得使 ![[公式]](https://www.zhihu.com/equation?tex=P%28x%3Ct%29%3Da) 成立的 ![[公式]](https://www.zhihu.com/equation?tex=t) ， ![[公式]](https://www.zhihu.com/equation?tex=t) 即可以视作从该分部中得到的一个采样结果。这里有两个前提：一是概率密度函数可积；第二个是累积分布函数有反函数。假如条件不成立怎么办呢？MCMC就登场了。

**动机二**

假如对于高维随机变量，比如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BR%7D%5E%7B50%7D) ，若每一维取100个点，则总共要取 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B100%7D) ，而已知宇宙的基本粒子大约有 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B87%7D) 个，对连续的也同样如此。因此MCMC可以解决“维数灾难”问题。

#### M-H采样



#### Gibbs采样



参考资料：https://zhuanlan.zhihu.com/p/37121528 MCMC——其实没太看懂





参考资料：https://cloud.tencent.com/developer/article/1102103（分类算法总结）

## 激活函数

### ☣️Relu Function

![img](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Ramp_function.svg/325px-Ramp_function.svg.png)



## 损失函数

### ☣️交叉熵损失函数





## 其他未分类

### ☣️拉格朗日松弛



### ☣️贪婪算法 



### ☣️网络流算法



### ☣️黑塞矩阵