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



## 近似算法

### ✅ 前言

**问题一**：航班选择最优，共计12个航班，要使得总的票价最少且每个人的等待时间之和最小

**问题二**：学生选择宿舍的问题，每个学生可以实现填报志愿，如果安排的宿舍与志愿完全一致，则代价为0，与第二志愿一致，代价为1，如果没有和志愿一致，代价为3。 

**核心**：将自己要优化的目标量化的表达出来，是解决优化函数的关键

**损失函数**：如在普通的数值优化问题中，可选择当前值与目标值距离的绝对之差，或者使用平方损失函数均可。

代码参考资料：https://blog.csdn.net/bcj296050240/article/details/50839806

### ✅ 随机搜索算法

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

### ✅ 爬山法（Hill Climbing）

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

### 🆘 模拟退火（Simulate Annealing）

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

### 🆘 遗传算法

> 选择（Select）：可理解为优胜劣汰，部分适应能力强的个体被保留，而较弱的则逐渐消失。常用的选择策略是 “比例选择”，也就是个体被选中的概率与其适应度函数值成正比。
>
> 交叉(Crossover)： 即繁衍后代，两个个体的DNA进行交叉融合，从而产生下一代。
>
> 变异(Mutation)：在繁殖过程中，下一代染色体的基因会有一定概率产生变异。变异发生的概率记为P。

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



### ☣️ 禁忌搜索（Taboo Search）

局部检索算法，





### ☣️ 蚁群算法



### ☣️ 粒子群算法（1995）

```python
# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import matplotlib.pyplot as plt

# 目标函数定义
def ras(x):
    y = 20 + x[0]**2 + x[1]**2 - 10*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))
    return y
    
# 参数初始化
w = 1.0
c1 = 1.49445
c2 = 1.49445

maxgen = 200   # 进化次数  
sizepop = 20   # 种群规模

# 粒子速度和位置的范围
Vmax =  1
Vmin = -1
popmax =  5
popmin = -5


# 产生初始粒子和速度
pop = 5 * np.random.uniform(-1,1,(2,sizepop))
v = np.random.uniform(-1,1,(2,sizepop))


fitness = ras(pop)             # 计算适应度
i = np.argmin(fitness)      # 找最好的个体
gbest = pop                    # 记录个体最优位置
zbest = pop[:,i]              # 记录群体最优位置
fitnessgbest = fitness        # 个体最佳适应度值
fitnesszbest = fitness[i]      # 全局最佳适应度值


# 迭代寻优
t = 0
record = np.zeros(maxgen)
while t < maxgen:
    
    # 速度更新
    v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * np.random.random() * (zbest.reshape(2,1) - pop)
    v[v > Vmax] = Vmax     # 限制速度
    v[v < Vmin] = Vmin
    
    # 位置更新
    pop = pop + 0.5 * v;
    pop[pop > popmax] = popmax  # 限制位置
    pop[pop < popmin] = popmin
    
    '''
    # 自适应变异
    p = np.random.random()             # 随机生成一个0~1内的数
    if p > 0.8:                          # 如果这个数落在变异概率区间内，则进行变异处理
        k = np.random.randint(0,2)     # 在[0,2)之间随机选一个整数
        pop[:,k] = np.random.random()  # 在选定的位置进行变异 
    '''

    # 计算适应度值
    fitness = ras(pop)
    
    # 个体最优位置更新
    index = fitness < fitnessgbest
    fitnessgbest[index] = fitness[index]
    gbest[:,index] = pop[:,index]

    # 群体最优更新
    j = np.argmin(fitness)
    if fitness[j] < fitnesszbest:
        zbest = pop[:,j]
        fitnesszbest = fitness[j]

    record[t] = fitnesszbest # 记录群体最优位置的变化   
    
    t = t + 1
    

# 结果分析
print zbest

plt.plot(record,'b-')
plt.xlabel('generation')  
plt.ylabel('fitness')  
plt.title('fitness curve')  
plt.show()
```

粒子群算法，也称粒子群优化算法或鸟群觅食算法（Particle Swarm Optimization），缩写为 PSO， 是近年来由J. Kennedy和R. C. Eberhart等开发的一种新的进化算法(Evolutionary Algorithm - EA)。PSO 算法属于进化算法的一种，和模拟退火算法相似，它也是从随机解出发，通过迭代寻找最优解，它也是通过适应度来评价解的品质，但它比遗传算法规则更为简单，它没有遗传算法的“交叉”(Crossover) 和“变异”(Mutation) 操作，它通过追随当前搜索到的最优值来寻找全局最优。这种算法以其实现容易、精度高、收敛快等优点引起了学术界的重视，并且在解决实际问题中展示了其优越性。粒子群算法是一种并行算法。

PSO模拟的是鸟群的捕食行为。设想这样一个场景：一群鸟在随机搜索食物。在这个区域里只有一块食物。所有的鸟都不知道食物在那里。但是他们知道当前的位置离食物还有多远。那么找到食物的最优策略是什么呢。最简单有效的就是搜寻目前离食物最近的鸟的周围区域。

参考：https://www.cnblogs.com/21207-iHome/p/6062535.html（源码）、https://www.jianshu.com/p/7e097bfb6390（通俗解释）

### ✅ 布谷鸟搜索（2009 Cuckoo Search）

<img src="https://vlight.me/images/CS_via_Levy_Flights.png" style="zoom:50%" />

觉得没啥很大用

参考资料：https://vlight.me/2017/12/17/Cuckoo-Search/

### 🆘 综合案例

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

