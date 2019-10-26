---

layout: post
title: "机器学习与数据挖掘算法"
subtitle: 'Machine Learning and Data Mining Algorithm'
author: "sean"
header-img: "img/post-bg-ai.jpeg"
tags:
  - 算法
  - 机器学习
  - 数据挖掘
---



> ✅  已经整理好
>
> 💤  大部分整理好了但是没理解
>
> 🆘  还有一部分需要整理
>
> ☣️  需要整理

备注：部分算法在多个分类中均有应用，可能只写在一个分类中，或者放在了回归模型中


## 机器学习与数据挖掘

### 决策树（Decision Tree）

#### 前言

**基本概念**

（1）根结点(Root Node)：它表示整个样本集合，并且该节点可以进一步划分成两个或多个子集。

（2）拆分(Splitting)：表示将一个结点拆分成多个子集的过程。

（3）决策结点(Decision Node)：当一个子结点进一步被拆分成多个子节点时，这个子节点就叫做决策结点。

（4）叶子结点(Leaf/Terminal Node)：无法再拆分的结点被称为叶子结点。

（5）剪枝(Pruning)：移除决策树中子结点的过程就叫做剪枝，跟拆分过程相反。

（6）分支/子树(Branch/Sub-Tree)：一棵决策树的一部分就叫做分支或子树。

（7）父结点和子结点(Paren and Child Node)：一个结点被拆分成多个子节点，这个结点就叫做父节点；其拆分后的子结点也叫做子结点。

**分类**

- 离散性决策树：离散性决策树，其目标变量是离散的，如性别：男或女等；
- 连续性决策树：连续性决策树，其目标变量是连续的，如工资、价格、年龄等；

![img](https://shuwoom.com/wp-content/uploads/2018/10/65e4301fe9e9a0f4e4d448a8f4181751.png)

**特征选择**

特征选择表示从众多的特征中选择一个特征作为当前节点分裂的标准，如何选择特征有不同的量化评估方法，从而衍生出不同的决策树，如ID3（通过信息增益选择特征）、C4.5（通过信息增益比选择特征）、CART（通过Gini指数选择特征）等。

**决策树的生成**

根据选择的特征评估标准，从上至下递归地生成子节点，直到数据集不可分则停止决策树停止生长。这个过程实际上就是使用满足划分准则的特征不断的将数据集划分成纯度更高，不确定行更小的子集的过程。对于当前数据集的每一次划分，都希望根据某个特征划分之后的各个子集的纯度更高，不确定性更小。

**决策树的裁剪**

决策树容易过拟合，一般需要剪枝来缩小树结构规模、缓解过拟合。

参考资料：https://shuwoom.com/?p=1452

#### ID3算法

熵这个概念最早起源于物理学，在物理学中是用来度量一个热力学系统的无序程度，而在信息学里面，熵是对不确定性的度量。在1948年，香农引入了信息熵（information entropy），将其定义为离散随机事件出现的概率，一个系统越是有序，信息熵就越低，反之一个系统越是混乱，它的信息熵就越高。所以信息熵可以被认为是系统有序化程度的一个度量。

**案例**：对于有K个类别的分类问题来说，假定样本集合D中第 k 类样本所占的比例为pk（k=1,2,...,K）,则样本集合D的信息熵定义为: 

$$Ent(D)=-\sum_{k=1}^KP_k·log_2p_k$$

**简单概括**：分叉树上的熵的加权平均值

```python
from math import log
import operator

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree


if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    print(createTree(dataSet, labels))  # 输出决策树模型结果
```

#### C4.5算法

**背景**：我们知道信息增益会偏向取值较多的特征，使用信息增益比可以对这一问题进行校正。

**信息增益**：$Gain(D,A) = H(D) – H(D|A)$，用于ID3的判断

**信息增益比**：特征A对训练数据集D的信息增益比GainRatio(D,A)定义为其信息增益Gain(D,A)与训练数据集D的经验熵H(D)之比：

$$GainRatio(D,A)=\frac{Gain(D,A)}{H(D)}$$

```python
def choose_best_feature_to_split(data_set):
    """
    按照最大信息增益比划分数据
    :param data_set: 样本数据，如： [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    :return:
    """
    num_feature = len(data_set[0]) - 1 # 特征个数，如：不浮出水面是否可以生存	和是否有脚蹼
    base_entropy = calc_shannon_ent(data_set) # 经验熵H(D)
    best_info_gain_ratio = 0.0
    best_feature_idx = -1
    for feature_idx in range(num_feature):
        feature_val_list = [number[feature_idx] for number in data_set]  # 得到某个特征下所有值（某列）
        unique_feature_val_list = set(feature_val_list)  # 获取无重复的属性特征值
        new_entropy = 0
        split_info = 0.0
        for value in unique_feature_val_list:
            sub_data_set = split_data_set(data_set, feature_idx, value)
            prob = len(sub_data_set) / float(len(data_set))  # 即p(t)
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 对各子集香农熵求和
            split_info += -prob * log(prob, 2)
        info_gain = base_entropy - new_entropy  # 计算信息增益，g(D,A)=H(D)-H(D|A)
        if split_info == 0:  # fix the overflow bug
            continue
        info_gain_ratio = info_gain / split_info
        # 最大信息增益比
        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_feature_idx = feature_idx

    return best_feature_idx
```

#### CART算法

**概述**：CART分类树预测分类离散型数据，采用基尼指数选择最优特征，同时决定该特征的最优二值切分点。

**概率分布的基尼指数定义**：假设有K个类，样本点属于第k个类的概率为Pk

$$Gini(p)=\sum_{k=1}^mp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2$$

**集合D的基尼指数**：

$$Gini(D)=1-\sum_{k=1}^K(\frac{|C_k|}{D})^2$$

**过程概述**：计算基尼系数的加权平均，求min

![img](https://shuwoom.com/wp-content/uploads/2018/10/fb6de7e8b17762b1caa14d6968407289.png)

![img](https://shuwoom.com/wp-content/uploads/2018/10/251158892a1340a1829e3a9f23dc477b.png)

```python
# -*- coding: utf-8 -*-
import numpy as np

class Tree(object):
    def __init__(self, value=None, true_branch=None, false_branch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data

    def __str__(self):
        print(self.col, self.value)
        print(self.results)
        print(self.summary)
        return ""

def split_datas(rows, value, column):
    """
    根据条件分离数据集
    :param rows:
    :param value:
    :param column:
    :return:  (list1, list2)
    """
    list1 = []
    list2 = []
    if isinstance(value, int) or isinstance(value, float):
        for row in rows:
            if row[column] >= value:
                list1.append(row)
            else:
                list2.append(row)
    else:
        for row in rows:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)

    return list1, list2


def calculate_diff_count(data_set):
    """
    分类统计data_set中每个类别的数量
    :param datas:如：[[5.1, 3.5, 1.4, 0.2, 'setosa'], [4.9, 3, 1.4, 0.2, 'setosa'],....]
    :return: 如：{'setosa': 50, 'versicolor': 50, 'virginica': 50}
    """
    results = {}
    for data in data_set:
        # 数据的最后一列data[-1]是类别
        if data[-1] not in results:
            results.setdefault(data[-1], 1)
        else:
            results[data[-1]] += 1
    return results


def gini(data_set):
    """
    计算gini的值，即Gini(p)
    :param data_set: 如：[[5.1, 3.5, 1.4, 0.2, 'setosa'], [4.9, 3, 1.4, 0.2, 'setosa'],....]
    :return:
    """
    length = len(data_set)
    category_2_cnt = calculate_diff_count(data_set)
    sum = 0.0
    for category in category_2_cnt:
        sum += pow(float(category_2_cnt[category]) / length, 2)
    return 1 - sum


def build_decision_tree(data_set, evaluation_function=gini):
    """
    递归建立决策树，当gain=0时，停止回归
    :param data_set: 如：[[5.1, 3.5, 1.4, 0.2, 'setosa'], [4.9, 3, 1.4, 0.2, 'setosa'],....]
    :param evaluation_function:
    :return:
    """
    current_gain = evaluation_function(data_set)
    column_length = len(data_set[0])
    rows_length = len(data_set)

    best_gain = 0.0
    best_value = None
    best_set = None

    # choose the best gain
    for feature_idx in range(column_length - 1):
        feature_value_set = set(row[feature_idx] for row in data_set)
        for feature_value in feature_value_set:
            sub_data_set1, sub_data_set2 = split_datas(data_set, feature_value, feature_idx)
            p = float(len(sub_data_set1)) / rows_length
            # Gini(D,A)表示在特征A的条件下集合D的基尼指数，gini_d_a越小，样本集合不确定性越小
            # 我们的目的是找到另gini_d_a最小的特征，及gain最大的特征
            gini_d_a = p * evaluation_function(sub_data_set1) + (1 - p) * evaluation_function(sub_data_set2)
            gain = current_gain - gini_d_a
            if gain > best_gain:
                best_gain = gain
                best_value = (feature_idx, feature_value)
                best_set = (sub_data_set1, sub_data_set2)
    dc_y = {'impurity': '%.3f' % current_gain, 'sample': '%d' % rows_length}

    # stop or not stop
    if best_gain > 0:
        true_branch = build_decision_tree(best_set[0], evaluation_function)
        false_branch = build_decision_tree(best_set[1], evaluation_function)
        return Tree(col=best_value[0], value=best_value[1], true_branch=true_branch, false_branch=false_branch, summary=dc_y)
    else:
        return Tree(results=calculate_diff_count(data_set), summary=dc_y, data=data_set)


def prune(tree, mini_gain, evaluation_function=gini):
    """
    裁剪
    :param tree:
    :param mini_gain:
    :param evaluation_function:
    :return:
    """
    if tree.true_branch.results == None:
        prune(tree.true_branch, mini_gain, evaluation_function)
    if tree.false_branch.results == None:
        prune(tree.false_branch, mini_gain, evaluation_function)

    if tree.true_branch.results != None and tree.false_branch.results != None:
        len1 = len(tree.true_branch.data)
        len2 = len(tree.false_branch.data)
        len3 = len(tree.true_branch.data + tree.false_branch.data)

        p = float(len1) / (len1 + len2)

        gain = evaluation_function(tree.true_branch.data + tree.false_branch.data) \
               - p * evaluation_function(tree.true_branch.data)\
               - (1 - p) * evaluation_function(tree.false_branch.data)

        if gain < mini_gain:
            # 当节点的gain小于给定的 mini Gain时则合并这两个节点
            tree.data = tree.true_branch.data + tree.false_branch.data
            tree.results = calculate_diff_count(tree.data)
            tree.true_branch = None
            tree.false_branch = None


def classify(data, tree):
    """
    分类
    :param data:
    :param tree:
    :return:
    """
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        else:
            if v == tree.value:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        return classify(data, branch)


def load_csv():
    def convert_types(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s
    data = np.loadtxt("datas.csv", dtype="str", delimiter=",")
    data = data[1:, :]
    data_set = ([[convert_types(item) for item in row] for row in data])
    return data_set


if __name__ == '__main__':
    data_set = load_csv()
    print data_set
    decistion_tree = build_decision_tree(data_set, evaluation_function=gini)
    print decistion_tree.results
    # prune(decistion_tree, 0.4)
    print classify([5.1,3.5,1.4,0.2], decistion_tree) # setosa
    print classify([6.8,2.8,4.8,1.4], decistion_tree) # versicolor
    print classify([6.8,3.2,5.9,2.3], decistion_tree) # virginica
```

### ☣️ KNN分类



### ☣️ PCA模型



### ☣️ EM算法



### ☣️ Apriori算法



### ☣️ 最大期望算法



### ☣️ K-Means算法



### ☣️ SVM算法



### ☣️ Adaboost算法



### ☣️ 贝叶斯模型（Naive Bayes）

```python
#!/usr/bin/python
# -*- coding: utf-8 -*-
#########################################
# Bayes : 用来描述两个条件概率之间的关系

# 参数:        inX: vector to compare to existing dataset (1xN)
#             dataSet: size m data set of known vectors (NxM)
#             labels: data set labels (1xM vector)
#             公式：P(A|B)=P(B|A)P(A)/P(B)
# 输出:       出错率
#########################################

import numpy as npy
import os
import time

#P(B|A)=P(A|B)*P(A)/P(B)

# 数据集目录
dataSetDir ='/Users/sean/Datasets'

class Bayes:
    def __init__(self):
        self.length=-1
        self.labelrate=dict()
        self.vectorrate=dict()

    def fit(self,dataset:list,labels:list):
        print("训练开始")
        if len(dataset)!=len(labels):
            raise ValueError("输入测试数组和类别数组长度不一致")
        self.length=len(dataset[0])#训练数据特征值的长度
        labelsnum=len(labels) #类别的数量
        norlabels=set(labels) #不重复类别的数量
        for item in norlabels:
            self.labelrate[item]=labels.count(item)/labelsnum #求当前类别占总类别的比例
        for vector,label in zip(dataset,labels):
            if label not in self.vectorrate:
                self.vectorrate[label]=[]
            self.vectorrate[label].append(vector)
        print("训练结束")
        return self

    def btest(self,testdata,labelset):
        if self.length==-1:
            raise ValueError("未开始训练，先训练")
        #计算testdata分别为各个类别的概率
        lbDict=dict()
        for thislb in labelset:
            p = 1
            alllabel = self.labelrate[thislb]
            allvector = self.vectorrate[thislb]
            vnum=len(allvector)
            allvector=npy.array(allvector).T
            for index in range(0,len(testdata)):
                vector=list(allvector[index])
                p*=vector.count(testdata[index])/vnum
            lbDict[thislb]=p * alllabel
        thislbabel=sorted(lbDict,key=lambda x:lbDict[x],reverse=True)[0]
        return thislbabel

#加载数据
def datatoarray(fname):
    arr=[]
    fh=open(fname)
    for i in range(0,32):
        thisline=fh.readline()
        for j in range(0 , 32):
            arr.append(int(thisline[j]))
    return arr

#建立一个函数取出labels
def seplabel(fname):
    filestr=fname.split(".")[0]
    label=int(filestr.split("_")[0])
    return label

#建立训练数据
def traindata():
    labels=[]
    trainfile=os.listdir(dataSetDir+"trainingDigits") # 加载测试数据
    num=len(trainfile)
    trainarr=npy.zeros((num,1024))
    for i in range(num):
        thisfname=trainfile[i]
        thislabel=seplabel(thisfname)
        labels.append(thislabel)
        trainarr[i,]=datatoarray(dataSetDir+"trainingDigits/"+thisfname)
    return trainarr,labels

# 贝叶斯算法手写识别主流程
bys=Bayes()
start = time.time()

# # step 1: 训练数据集
train_data,labels=traindata()
train_data=list(train_data)
bys.fit(train_data,labels)

# # step 2:测试数据集
thisdata=datatoarray(dataSetDir+"testDigits/8_90.txt")
labelsall=[0,1,2,3,4,5,6,7,8,9]

# # 识别单个手写体数字
# test=bys.btest(thisdata,labelsall)
# print(test)

# # 识别多个手写体数字（批量处理）,并输出结果
testfile=os.listdir(dataSetDir+"testDigits")
num=len(testfile)
x=0
for i in range(num):
    thisfilename=testfile[i]
    thislabel=seplabel(thisfilename)
    thisdataarr=datatoarray(dataSetDir+"testDigits/"+thisfilename)
    label=bys.btest(thisdataarr,labelsall)
    print("测试数字是："+str(thislabel)+"  识别出来的数字是："+str(label))
    if label!=thislabel:
        x+=1
        print("识别出错")
print(x)
print("出错率："+str(x/num))

end = time.time()
running_time = end-start
print('程序运行总耗时： %.5f sec' %running_time)
```

```flow
st=>start: 开始
op=>operation: My Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
```

## 相关内容

### 归一化

归一化一般是将数据映射到指定的范围，用于去除不同维度数据的量纲以及量纲单位。

常见的映射范围有 [0, 1] 和 [-1, 1] ，最常见的归一化方法就是 **Min-Max 归一化**：

![[公式]](https://www.zhihu.com/equation?tex=x_{new}%3D\frac{x-x_{min}}{x_{max}-x_{min}})

举个例子，我们判断一个人的身体状况是否健康，那么我们会采集人体的很多指标，比如说：身高、体重、红细胞数量、白细胞数量等。

一个人身高 180cm，体重 70kg，白细胞计数 $7.50\times10^9/L$ ，etc.

衡量两个人的状况时，白细胞计数就会起到主导作用从而**遮盖住其他的特征**，归一化后就不会有这样的问题。 

参考：https://zhuanlan.zhihu.com/p/29957294

### **标准化 Normalization**

> 归一化和标准化的英文翻译是一致的，但是根据其用途（或公式）的不同去理解（或翻译）

下面我们将探讨最常见的标准化方法： **Z-Score 标准化**。

#### **Z-Score 标准化**

![[公式]](https://www.zhihu.com/equation?tex=x_{new}%3D\frac{x-\mu+}{\sigma+})

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmu) 是样本数据的**均值（mean）**， ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 是样本数据的**标准差（std）**。

![img](https://tva1.sinaimg.cn/large/006y8mN6gy1g8938kwyxgj30k0053749.jpg)



上图则是一个散点序列的标准化过程：原图->减去均值->除以标准差。

显而易见，变成了一个**均值为 0 ，方差为 1 的分布**，下图通过 Cost 函数让我们更好的理解标准化的作用。

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8938e6pv9j30k00b9q3g.jpg" alt="img" style="zoom:80%;" />



机器学习的目标无非就是不断优化损失函数，使其值最小。在上图中$J(w,b)$就是我们要优化的目标函数

我们不难看出，**标准化后可以更加容易地得出最优参数$w$和$b$以及计算出 $J(w,b)$ 的最小值，从而达到加速收敛的效果**。 ![[公式]](https://www.zhihu.com/equation?tex=%5E%7B%5B1%5D%7D)

*注：上图来源于 Andrew Ng 的课程讲义*

#### **Batch Normalization**

在机器学习中，**最常用标准化的地方**莫过于神经网络的 **BN 层（Batch Normalization）**，因此我们简单的谈谈 BN 层的原理和作用，想要更深入的了解可以[查看论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1502.03167)。

我们知道数据预处理做标准化可以加速收敛，同理，在神经网络使用标准化也可以**加速收敛**，而且还有如下好处：

- 具有正则化的效果（Batch Normalization reglarizes the model）
- 提高模型的泛化能力（Be advantageous to the generalization of network）
- 允许更高的学习速率从而加速收敛（Batch Normalization enables higher learning rates）

其原理是**利用正则化减少内部相关变量分布的偏移（Reducing Internal Covariate Shift）**，从而**提高了算法的鲁棒性**。 ![[公式]](https://www.zhihu.com/equation?tex=%5E%7B%5B2%5D%7D)

Batch Normalization 由两部分组成，第一部分是**缩放与平移（scale and shift）**，第二部分是**训练缩放尺度和平移的参数（train a BN Network）**，算法步骤如下：

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g8938vogvcj30k00gcmyg.jpg" alt="img" style="zoom:50%;" />

接下来训练 BN 层参数 $\gamma$ 和  $\beta$ ，限于篇幅的原因按下不表，有兴趣的读者可以拜读[这篇论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1502.03167)。

参考：https://zhuanlan.zhihu.com/p/29957294

### 正则处理

监督机器学习一般可以用一个公式来代表，右侧就是正则处理，为了防止过拟合
$$w^∗=arg_wmin∑_iL(yi,f(x_i;w))+λΩ(w)$$

其中，第 1 项是**经验风险**，第 2 项是**正则项**， ![[公式]](https://www.zhihu.com/equation?tex=%CE%BB%E2%89%A50) 为调整两者之间关系的系数。

第 1 项的经验风险较小的模型可能较复杂（有多个非零参数），这时第 2 项的模型复杂度会较大。

常见的有正则项有 **L1 正则** 和 **L2 正则** ，其中 **L2 正则** 的控制过拟合的效果比 **L1 正则** 的好。

**正则化的作用是选择经验风险与模型复杂度同时较小的模型**。 

常见的有正则项有 **L1 正则** 和 **L2 正则** 以及 **Dropout** ，其中 **L2 正则** 的控制过拟合的效果比 **L1 正则** 的好。

- L1正则化可以使得参数稀疏化，即得到的参数是一个稀疏矩阵，可以用于特征选择。
  - **稀疏性**，说白了就是模型的很多参数是0。通常机器学习中特征数量很多，例如文本处理时，如果将一个词组（term）作为一个特征，那么特征数量会达到上万个（bigram）。在预测或分类时，那么多特征显然难以选择，但是如果代入这些特征得到的模型是一个稀疏模型，很多参数是0，表示只有少数特征对这个模型有贡献，绝大部分特征是没有贡献的，即使去掉对模型也没有什么影响，此时我们就可以只关注系数是非零值的特征。这相当于对模型进行了一次特征选择，只留下一些比较重要的特征，提高模型的泛化能力，降低过拟合的可能。
- L2正则化可以防止模型过拟合（overfitting）；一定程度上，L1也可以防止过拟合

参考：https://zhuanlan.zhihu.com/p/29957294、https://www.cnblogs.com/zingp/p/10375691.html

#### **L1范数**

$$||X||_0=∑^n_{i=0}|X_i|$$

![[公式]](https://www.zhihu.com/equation?tex=J(w%2Cb)%3D\frac{1}{m}+\sum_{i%3D1}^{m}L(\hat{y}%2Cy)%2B\frac{\lambda+}{m}\left+\|+w+\right+\|_{1})

L1 范数是向量中各个元素绝对值之和，也被称作“Lasso regularization”（稀疏规则算子）。

凸函数，不是处处可微分

得到的是稀疏解（最优解常出现在顶点上，且顶点上的 w 只有很少的元素是非零的）

#### L2范数

$$||X||_2=\sqrt{∑^n_{i=0}X^2_i}$$

![[公式]](https://www.zhihu.com/equation?tex=J(w%2Cb)%3D\frac{1}{m}+\sum_{i%3D1}^{m}L(\hat{y}%2Cy)%2B\frac{\lambda+}{m}\left+\|+w+\right+\|_{1})

Euclid范数（欧几里得范数，常用计算向量长度），即向量元素绝对值的平方和再开方。 

- 凸函数，处处可微分
- 易于优化

### 激活函数

#### Relu

![这里写图片描述](https://img-blog.csdn.net/20151015103947207)

Relu激活函数的优点在于：

1. 梯度不饱和。梯度计算公式为：。因此在反向传播过程中，减轻了梯度弥散的问题，神经网络前几层的参数也可以很快的更新。
2. 计算速度快。正向传播过程中，sigmoid和tanh函数计算激活值时需要计算指数，而Relu函数仅需要设置阈值。如果，如果。加快了正向传播的计算速度。

因此，Relu激活函数可以极大地加快收敛速度，相比tanh函数，收敛速度可以加快6倍

#### Tanh

$$tanh(x)=2sigmoid(2x)-1$$

$$tanhx=\frac{sinhx}{coshx}=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

![img](https://upload-images.jianshu.io/upload_images/1531909-bee83ab606daa6e5.png?imageMogr2/auto-orient/strip|imageView2/2/w/494)

tanh函数将一个实数输入映射到[-1,1]范围内，如上图（右）所示。当输入为0时，tanh函数输出为0，符合我们对激活函数的要求。然而，tanh函数也存在梯度饱和问题，导致训练效率低下。

#### Sigmoid

<img src="/Users/sean/Library/Application Support/typora-user-images/image-20191023195904166.png" alt="image-20191023195904166" style="zoom:50%;" />

sigmoid将一个实数输入映射到[0,1]范围内，使用sigmoid作为激活函数存在以下几个问题：

1. 梯度饱和。当函数激活值接近于0或者1时，函数的梯度接近于0。在反向传播计算梯度过程中：，每层残差接近于0，计算出的梯度也不可避免地接近于0。这样在参数微调过程中，会引起参数弥散问题，传到前几层的梯度已经非常靠近0了，参数几乎不会再更新。
2. 函数输出不是以0为中心的。我们更偏向于当激活函数的输入是0时，输出也是0的函数。
3. 因为上面两个问题的存在，导致参数收敛速度很慢，严重影响了训练的效率。因此在设计神经网络时，很少采用sigmoid激活函数。

### 梯度问题

#### 梯度爆炸

原文：https://zhuanlan.zhihu.com/p/32154263

**梯度爆炸是什么？**

误差梯度在网络训练时被用来得到网络参数更新的方向和幅度，进而在正确的方向上以合适的幅度更新网络参数。在深层网络或递归神经网络中，**误差梯度在更新中累积得到一个非常大的梯度，这样的梯度会大幅度更新网络参数，进而导致网络不稳定**。**在极端情况下，权重的值变得特别大，以至于结果会溢出**（NaN值，[无穷与非数值](https://link.zhihu.com/?target=https%3A//baike.baidu.com/item/nan/7455322%3Ffr%3Daladdin)）。当梯度爆炸发生时，网络层之间反复乘以大于1.0的梯度值使得梯度值成倍增长。

**梯度爆炸会引发哪些问题？**

在深度多层感知机网络中，梯度爆炸会导致网络不稳定，最好的结果是无法从训练数据中学习，最坏的结果是由于权重值为NaN而无法更新权重。

*梯度爆炸会使得学习不稳定；*

—— [深度学习](https://link.zhihu.com/?target=http%3A//amzn.to/2fwdoKR)第282页

在循环神经网络（RNN）中，梯度爆炸会导致网络不稳定，使得网络无法从训练数据中得到很好的学习，最好的结果是网络不能在长输入数据序列上学习。

*梯度爆炸问题指的是训练过程中梯度大幅度增加，这是由于长期组件爆炸造成的；*

——[训练循环神经网络中的困难](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v28/pascanu13.pdf)

**如何知道网络中是否有梯度爆炸问题？**

在网络训练过程中，如果发生梯度爆炸，那么会有一些明显的迹象表明这一点，例如：

- 模型无法在训练数据上收敛（比如，损失函数值非常差）；
- 模型不稳定，在更新的时候损失有较大的变化；
- 模型的损失函数值在训练过程中变成NaN值； 

如果你遇到上述问题，我们就可以深入分析网络是否存在梯度爆炸问题。还有一些不太为明显的迹象可以用来确认网络中是否存在梯度爆炸问题：

- 模型在训练过程中，权重变化非常大；
- 模型在训练过程中，权重变成NaN值； 
- 每层的每个节点在训练时，其误差梯度值一直是大于1.0； 

**如何解决梯度爆炸问题？**

解决梯度爆炸问题的方法有很多，本部分将介绍一些有效的实践方法：

**1.重新设计网络模型**

在深层神经网络中，梯度爆炸问题可以通过将网络模型的层数变少来解决。此外，在训练网络时，使用较小批量也有一些好处。在循环神经网络中，训练时使用较小时间步长更新（也被称作[截断反向传播](https://link.zhihu.com/?target=https%3A//machinelearningmastery.com/gentle-introduction-backpropagation-time/)）可能会降低梯度爆炸发生的概率。

**2.使用修正线性激活函数**

在深度多层感知机中，当激活函数选择为一些之前常用的Sigmoid或Tanh时，网络模型会发生梯度爆炸问题。而使用修正线性激活函数（ReLU）能够减少梯度爆炸发生的概率，对于隐藏层而言，使用修正线性激活函数（ReLU）是一个比较合适的激活函数，当然ReLU函数有许多变体，大家在实践过程中可以逐一使用以找到最合适的激活函数。

**3.使用长短周期记忆网络**

由于循环神经网络中存在的固有不稳定性，梯度爆炸可能会发生。比如，通过时间反向传播，其本质是将循环网络转变为深度多层感知神经网络。通过使用长短期记忆单元（LSTM）或相关的门控神经结构能够减少梯度爆炸发生的概率。

对于循环神经网络的时间序列预测而言，采用LSTM是新的最佳实践。

**4.使用梯度裁剪**

在深度多层感知网络中，当有大批量数据以及LSTM是用于很长时间序列时，梯度爆炸仍然会发生。当梯度爆炸发生时，可以在网络训练时检查并限制梯度的大小，这被称作梯度裁剪。

*梯度裁剪是处理梯度爆炸问题的一个简单但非常有效的解决方案，如果梯度值大于某个阈值，我们就进行梯度裁剪。*

——[自然语言处理中的神经网络方法的第5.2.4节](https://link.zhihu.com/?target=http%3A//amzn.to/2fwTPCn)

具体而言，检查误差梯度值就是与一个阈值进行比较，若误差梯度值超过设定的阈值，则截断或设置为阈值。

*在某种程度上，梯度爆炸问题可以通过梯度裁剪来缓解（在执行梯度下降步骤之前对梯度进行阈值操作）*

——深度学习第294页

在Keras深度学习库中，在训练网络之前，可以对优化器的clipnorm和 clipvalue参数进行设置来使用梯度裁剪，一般而言，默认将clipnorm和 clipvalue分别设置为1和0.5.

- [在Keras API中使用优化器](https://link.zhihu.com/?target=https%3A//keras.io/optimizers/)

**5.使用权重正则化**

如果梯度爆炸问题仍然发生，另外一个方法是对网络权重的大小进行校验，并对大权重的损失函数增添一项惩罚项，这也被称作权重正则化，常用的有L1（权重的绝对值和）正则化与L2（权重的绝对值平方和再开方）正则化。

*使用L1或L2惩罚项会减少梯度爆炸的发生概率*

——[训练循环神经网络中的困难](https://link.zhihu.com/?target=http%3A//proceedings.mlr.press/v28/pascanu13.pdf%3Fspm%3D5176.100239.blogcont292826.13.57KVN0%26file%3Dpascanu13.pdf)

在Keras深度学习库中，可以在每层上使用L1或L2正则器设置kernel_regularizer参数来完成权重的正则化操作。

- [在Keras API中使用正则器](https://link.zhihu.com/?target=https%3A//keras.io/regularizers/)

#### 梯度消失

误差反向传播（BP）算法中为什么会产生梯度消失？ - 维吉特伯的回答 - 知乎
https://www.zhihu.com/question/49812013/answer/148825073

### 损失函数（Loss Function）

#### 交叉熵损失函数

$$C=-\frac{1}{n}\sum_x[y\,ln\,a+(1-y)ln(1-a)]$$

参考资料：https://blog.csdn.net/u014313009/article/details/51043064

损失函数（loss function）是用来**估量模型的预测值f(x)与真实值Y的不一致程度**，它是一个非负实值函数,通常使用L(Y, f(x))来表示，损失函数越小，模型的鲁棒性就越好。损失函数是**经验风险函数**的核心部分，也是**结构风险函数**重要组成部分。模型的结构风险函数包括了经验风险项和正则项，通常可以表示成如下式子：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/vchxtyghby.png?imageView2/2/w/1620)

　　其中，前面的均值函数表示的是经验风险函数，L代表的是损失函数，后面的Φ是正则化项（regularizer）或者叫惩罚项（penalty term），它可以是L1，也可以是L2，或者其他的正则函数。整个式子表示的意思是**找到使目标函数最小时的θ值**。下面主要列出几种常见的损失函数。

 理解：损失函数旨在表示出logit和label的差异程度，不同的损失函数有不同的表示意义，也就是在最小化损失函数过程中，logit逼近label的方式不同，得到的结果可能也不同。

一般情况下，softmax和sigmoid使用交叉熵损失（logloss），hingeloss是SVM推导出的，hingeloss的输入使用原始logit即可。

#### 二次代价函数

$$C=\frac{1}{2n}\sum_x||y(x)-a^L(x)||$$

$$\frac{\partial C}{\partial w}=(a-y)\sigma'(z)x $$

$$\frac{\partial C}{\partial b}=(a-y)\sigma'(z)$$

下文参考资料：https://cloud.tencent.com/developer/article/1165263

其他参考：https://www.csuldw.com/2016/03/26/2016-03-26-loss-function/

#### LogLoss对数损失函数（逻辑回归，交叉熵损失）

　　有些人可能觉得逻辑回归的损失函数就是平方损失，其实并不是。**平方损失函数可以通过线性回归在假设样本是高斯分布的条件下推导得到**，而逻辑回归得到的并不是平方损失。在逻辑回归的推导中，它假设样本服从**伯努利分布（0-1分布）**，然后求得满足该分布的似然函数，接着取对数求极值等等。而逻辑回归并没有求似然函数的极值，而是把极大化当做是一种思想，进而推导出它的经验风险函数为：**最小化负的似然函数（即max F(y, f(x)) —> min -F(y, f(x)))**。从损失函数的视角来看，它就成了log损失函数了。

**log损失函数的标准形式**：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/mblanzbpz4.png?imageView2/2/w/1620)

　　刚刚说到，取对数是为了方便计算极大似然估计，因为在MLE（最大似然估计）中，直接求导比较困难，所以通常都是先取对数再求导找极值点。损失函数L(Y, P(Y|X))表达的是样本X在分类Y的情况下，使概率P(Y|X)达到最大值（换言之，**就是利用已知的样本分布，找到最有可能（即最大概率）导致这种分布的参数值；或者说什么样的参数才能使我们观测到目前这组数据的概率最大**）。因为log函数是单调递增的，所以logP(Y|X)也会达到最大值，因此在前面加上负号之后，最大化P(Y|X)就等价于最小化L了。

　　逻辑回归的P(Y=y|x)表达式如下（为了将类别标签y统一为1和0，下面将表达式分开表示）：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/3uwkib8pna.png?imageView2/2/w/1620)

　　将它带入到上式，通过推导可以得到logistic的损失函数表达式，如下：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/clufn12j5o.png?imageView2/2/w/1620)

　　逻辑回归最后得到的目标式子如下：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/ywnixwm44c.png?imageView2/2/w/1620)

　　上面是针对二分类而言的。这里需要解释一下：**之所以有人认为逻辑回归是平方损失，是因为在使用梯度下降来求最优解的时候，它的迭代式子与平方损失求导后的式子非常相似，从而给人一种直观上的错觉**。

这里有个PDF可以参考一下：[Lecture 6: logistic regression.pdf](https://www.cs.berkeley.edu/~russell/classes/cs194/f11/lectures/CS194 Fall 2011 Lecture 06.pdf).

 **注意：softmax使用的即为交叉熵损失函数，binary_cossentropy为二分类交叉熵损失，categorical_crossentropy为多分类交叉熵损失，当使用多分类交叉熵损失函数时，标签应该为多分类模式，即使用one-hot编码的向量。**

#### 平方损失函数（最小二乘法, Ordinary Least Squares ）

　　最小二乘法是线性回归的一种，最小二乘法（OLS）将问题转化成了一个凸优化问题。在线性回归中，它假设样本和噪声都服从高斯分布（为什么假设成高斯分布呢？其实这里隐藏了一个小知识点，就是**中心极限定理**，可以参考[【central limit theorem】](https://en.wikipedia.org/wiki/Central_limit_theorem)），最后通过极大似然估计（MLE）可以推导出最小二乘式子。最小二乘的基本原则是：**最优拟合直线应该是使各点到回归直线的距离和最小的直线，即平方和最小**。换言之，OLS是基于距离的，而这个距离就是我们用的最多的**欧几里得距离**。为什么它会选择使用欧式距离作为误差度量呢（即Mean squared error， MSE），主要有以下几个原因：

- 简单，计算方便；
- 欧氏距离是一种很好的相似性度量标准；
- 在不同的表示域变换后特征性质不变。

**平方损失（Square loss）的标准形式如下：**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/0a6qmutmgz.png?imageView2/2/w/1620)

当样本个数为n时，此时的损失函数变为：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/yijq5f7pf6.png?imageView2/2/w/1620)

`Y-f(X)`表示的是残差，整个式子表示的是**残差的平方和**，而我们的目的就是最小化这个目标函数值（注：该式子未加入正则项），也就是**最小化残差的平方和（residual sum of squares，RSS）**。

而在实际应用中，通常会使用均方差（MSE）作为一项衡量指标，公式如下：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/40i7usorao.png?imageView2/2/w/1620)

上面提到了线性回归，这里额外补充一句，我们通常说的线性有两种情况，一种是因变量y是自变量x的线性函数，一种是因变量y是参数α的线性函数。在机器学习中，通常指的都是后一种情况。

#### 指数损失函数（Adaboost）

学过Adaboost算法的人都知道，它是前向分步加法算法的特例，是一个加和模型，损失函数就是指数函数。在Adaboost中，经过m此迭代之后，可以得到fm(x):

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/9csxlzc8kc.png?imageView2/2/w/1620)

Adaboost每次迭代时的目的是为了找到最小化下列式子时的参数α 和G：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/s5jxlbyyu9.png?imageView2/2/w/1620)

**而指数损失函数（exp-loss）的标准形式如下**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/1w1scqkvld.png?imageView2/2/w/1620)

可以看出，Adaboost的目标式子就是指数损失，在给定n个样本的情况下，Adaboost的损失函数为：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/rcsuvb3aom.png?imageView2/2/w/1620)

关于Adaboost的推导，可以参考Wikipedia：[AdaBoost](https://en.wikipedia.org/wiki/AdaBoost)或者《统计学习方法》P145.

#### Hinge损失函数（SVM）

在机器学习算法中，hinge损失函数和SVM是息息相关的。在**线性支持向量机**中，最优化问题可以等价于下列式子：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/jpaf847fuj.png?imageView2/2/w/1620)

下面来对式子做个变形，令：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/fof2ryr62o.png?imageView2/2/w/1620)

于是，原式就变成了：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/dwnvocjbj0.png?imageView2/2/w/1620)

如若取λ=1/(2C)，式子就可以表示成： 

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/fr0tcbmweo.png?imageView2/2/w/1620)

可以看出，该式子与下式非常相似：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/gwesvh36xu.png?imageView2/2/w/1620)

前半部分中的 l 就是hinge损失函数，而后面相当于L2正则项。

**Hinge 损失函数的标准形式**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/9wh4maztst.png?imageView2/2/w/1620)

可以看出，当|y|>=1时，L(y)=0。

更多内容，参考[Hinge-loss](https://en.wikipedia.org/wiki/Hinge_loss)。

补充一下：在libsvm中一共有4中核函数可以选择，对应的是`-t`参数分别是：

- 0-线性核；
- 1-多项式核；
- 2-RBF核；
- 3-sigmoid核。

#### 其它损失函数

除了以上这几种损失函数，常用的还有：

**0-1损失函数**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/j8j9f45t58.png?imageView2/2/w/1620)

**绝对值损失函数**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/pq3irn4nn6.png?imageView2/2/w/1620)

下面来看看几种损失函数的可视化图像，对着图看看横坐标，看看纵坐标，再看看每条线都表示什么损失函数，多看几次好好消化消化。

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/nkhl6fmtst.png?imageView2/2/w/1620)

### 优化器

#### Batch Gradient Descent （BGD）

**梯度更新规则:**

BGD 采用整个训练集的数据来计算 cost function 对参数的梯度：

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/83gnmlzeqe.png?imageView2/2/w/1620)

**缺点：**

**由于这种方法是在一次更新中，就对整个数据集计算梯度，所以计算起来非常慢，遇到很大量的数据集也会非常棘手，而且不能投入新数据实时更新模型。**

```javascript
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

我们会事先定义一个迭代次数 epoch，首先计算梯度向量 params_grad，然后沿着梯度的方向更新参数 params，learning rate 决定了我们每一步迈多大。

**Batch gradient descent 对于凸函数可以收敛到全局极小值，对于非凸函数可以收敛到局部极小值。**

#### Stochastic Gradient Descent (SGD)

**梯度更新规则:**

和 BGD 的一次用所有数据计算梯度相比，SGD 每次更新时对每个样本进行梯度更新，对于很大的数据集来说，可能会有相似的样本，这样 BGD 在计算梯度时会出现冗余，而 **SGD 一次只进行一次更新，就没有冗余，而且比较快，并且可以新增样本。**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/oa4bsslnd8.png?imageView2/2/w/1620)

```javascript
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

 看代码，可以看到区别，就是整体数据集是个循环，其中对每个样本进行一次参数更新。

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/y5mhqxi8vr.png?imageView2/2/w/1620)

随机梯度下降是通过每个样本来迭代更新一次，如果样本量很大的情况，那么可能只用其中部分的样本，就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次。**缺点是SGD的噪音较BGD要多，使得SGD并不是每次迭代都向着整体最优化方向**。**所以虽然训练速度快，但是准确度下降，并不是全局最优**。**虽然包含一定的随机性，但是从期望上来看，它是等于正确的导数的。**

**缺点：**

**SGD 因为更新比较频繁，会造成 cost function 有严重的震荡。**

**BGD 可以收敛到局部极小值，当然 SGD 的震荡可能会跳到更好的局部极小值处。**

**当我们稍微减小 learning rate，SGD 和 BGD 的收敛性是一样的。**

#### Mini-Batch Gradient Descent （MBGD）

**梯度更新规则：**

**MBGD 每一次利用一小批样本，即 n 个样本进行计算，这样它可以降低参数更新时的方差，收敛更稳定，另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算。**

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/755zvm0j1o.png?imageView2/2/w/1620)

和 SGD 的区别是每一次循环不是作用于每个样本，而是具有 n 个样本的批次。

```javascript
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

 **超参数设定值:  n 一般取值在 50～256**

**缺点：（两大缺点）**

1. **不过 Mini-batch gradient descent 不能保证很好的收敛性，learning rate 如果选择的太小，收敛速度会很慢，如果太大，loss function 就会在极小值处不停地震荡甚至偏离。（有一种措施是先设定大一点的学习率，当两次迭代之间的变化低于某个阈值后，就减小 learning rate，不过这个阈值的设定需要提前写好，这样的话就不能够适应数据集的特点。）**对于非凸函数，还要避免陷于局部极小值处，或者鞍点处，因为鞍点周围的error是一样的，所有维度的梯度都接近于0，SGD 很容易被困在这里。（**会在鞍点或者局部最小点震荡跳动，因为在此点处，如果是训练集全集带入即BGD，则优化会停止不动，如果是mini-batch或者SGD，每次找到的梯度都是不同的，就会发生震荡，来回跳动。**）
2. SGD对所有参数更新时应用同样的 learning rate，如果我们的数据是稀疏的，**我们更希望对出现频率低的特征进行大一点的更新。LR会随着更新的次数逐渐变小。**

鞍点就是：一个光滑函数的鞍点邻域的曲线，曲面，或超曲面，都位于这点的切线的不同边。例如这个二维图形，像个马鞍：在x-轴方向往上曲，在y-轴方向往下曲，鞍点就是（0，0）。

![img](https://ask.qcloudimg.com/http-save/yehe-1881084/2cxevj8bpb.png?imageView2/2/w/1620)

#### 其他优化算法

- [ ] Momentum
- [ ] Nesterov Accelerated Gradient
- [ ] Adagrad （Adaptive gradient algorithm）
- [ ] Adadelta
- [ ] RMSprop
- [ ] Adam：Adaptive Moment Estimation

#### 如何选择优化算法

**如果数据是稀疏的，就用自适用方法，即 Adagrad, Adadelta, RMSprop, Adam。**

**RMSprop, Adadelta, Adam 在很多情况下的效果是相似的。**

**Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum，**

**随着梯度变的稀疏，Adam 比 RMSprop 效果会好。**

整体来讲，**Adam 是最好的选择**。

很多论文里都会用 SGD，没有 momentum 等。**SGD 虽然能达到极小值，但是比其它算法用的时间长，而且可能会被困在鞍点**。

如果需要更快的收敛，或者是训练更深更复杂的神经网络，需要用一种自适应的算法。

原文：https://cloud.tencent.com/developer/article/1118673

### Dropout

原文：https://blog.csdn.net/stdcoutzyx/article/details/49022443、https://zhuanlan.zhihu.com/p/38200980

大规模的神经网络有两个缺点：

- 费时
- 容易过拟合

这两个缺点真是抱在深度学习大腿上的两个大包袱，一左一右，相得益彰，额不，臭气相投。过拟合是很多机器学习的通病，过拟合了，得到的模型基本就废了。而为了解决过拟合问题，一般会采用ensemble方法，即训练多个模型做组合，此时，费时就成为一个大问题，不仅训练起来费时，测试起来多个模型也很费时。总之，几乎形成了一个死锁。

Dropout的出现很好的可以解决这个问题，每次做完dropout，相当于从原始的网络中找到一个更瘦的网络，如下图所示：

![img](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n7-1.png)

因而，对于一个有N个节点的神经网络，有了dropout后，就可以看做是2n个模型的集合了，但此时要训练的参数数目却是不变的，这就解脱了费时的问题。

dropout带来的模型的变化

而为了达到ensemble的特性，有了dropout后，神经网络的训练和预测就会发生一些变化。

训练层面

![img2](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n7-5.png)

无可避免的，训练网络的每个单元要添加一道概率流程。 


对应的公式变化如下如下：

没有dropout的神经网络

![img3](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n7-3.png) 

有dropout的神经网络 

![img4](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n7-4.png)

测试层面

预测的时候，每一个单元的参数要预乘以p。 

### ![img5](https://raw.githubusercontent.com/stdcoutzyx/Blogs/master/blogs/imgs/n7-2.png)区分Loss/Cost/Object函数

Loss Function 是定义在单个样本上的，算的是一个样本的误差。

Cost Function 是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均。

Object Function（目标函数 ）定义为：Cost Function + 正则化项。

## 前沿研究

### Sequence to Sequence

![这里写图片描述](https://img-blog.csdn.net/20171101103708217)

![img](https://images2018.cnblogs.com/blog/1192699/201808/1192699-20180806141638641-1078830067.png)

**概述**：一种函数映射模型，具有弱约束的多对多对应关系。

**原理**：左侧Encoder编码将输入序列转化成一个固定长度的向量编码，右侧Decoder解码将之前生成的固定向量再转化成输出序列，编解码部分可以采用CNN、RNN、LSTM、GRU、BLSTM等实现。可以预测任意的序列对应关系。

**意义**：第一次提出了Encoder-Decoder模型，主要用于序列预测

**问题**：Encoder是将输入内容压缩到一个固定长度的向量中，语义编码是固定长度，然后计算出内容进行Decoder

### Attention Mechanism

《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型，基于一个Encoder和一个Decoder来构建基于神经网络的End-to-End的机器翻译模型，其中，Encoder把输入X编码成一个固定长度的隐向量Z，Decoder基于隐向量Z解码出目标输出Y。这是一个非常经典的序列到序列的模型，但是却存在**两个明显的问题**：

1、把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。

2、把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往是模型性能下降。

**模型示例**：可以与Seq2Seq对比，中间增加了一个权重层，通过不同的语义编码实现（也就是不同长度的向量）

<img src="https://images2018.cnblogs.com/blog/1192699/201808/1192699-20180806142115912-1682939089.png" alt="img" style="zoom:70%;" />

<img src="https://images2018.cnblogs.com/blog/1192699/201808/1192699-20180806142159178-1634092293.png" alt="img" style="zoom:70%;" />

参考资料：https://zhuanlan.zhihu.com/p/31547842（注意力模型汇总机及其应用）、https://www.cnblogs.com/guoyaohua/p/9429924.html（注意力机制解释及其应用）

### ☣️ Pointer Network



### ☣️ Graph Embedding



##迁移学习





## 强化学习（Reinforcement Learning）

### ☣️ 介绍

![img](https://user-gold-cdn.xitu.io/2019/8/18/16ca41b0a58edcee?imageView2/0/w/1280/h/960/ignore-error/1)



如上图左边所示，一个agent(例如：玩家/智能体等)做出了一个action，对environment造成了影响，也就是改变了state，而environment为了反馈给agent，agent就得到了一个奖励(例如：积分/分数)，不断的进行这样的循环，直到结束为止。

上述过程就相当于一个**马尔可夫决策过程**

上图右边所示，S0 状态经过了 a0 的行为后，获得了奖励 r1 ，变成了状态S1，后又经过了 a0 行为得到奖励 r2，变成了状态 S2 ，如此往复循环，直到结束为止。

通过以上的描述，大家都已经确定了一个概念，也就是agent(智能体)在当下做出的决定肯定使得未来收益最大化，那么，一个马儿可夫决策过程对应的奖励总和为：

$$R=r_1+r_2+r_3+...+r_n$$

t 时刻(当下)的未来奖励，只考虑后面的奖励，前面的改变不了：

$R_t=r_t+r_{t+1}+r_{t+2}+...+r_n$

当前的行为对于未来是不确定性的，要打一个折扣，也就是加入一个系数gamma，是一个 0 到 1 的值：

$$R_t=r_1+\gamma_{}r_{t+1}+\gamma^2r_{t+2}+...+\gamma^{n-1}r_n$$

离当前越远的时间，gamma的惩罚系数就会越大，也就是越不确定。为的就是在当前和未来的决策中取得一个平衡。gamma取 0 ，相当于不考虑未来，只考虑当下，是一种很短视的做法；而gamma取 1 ，则完全考虑了未来，又有点过虑了。所以一般gamma会取 0 到 1 之间的一个值。

Rt 可以用 Rt+1 来表示，写成递推式：

$$R_t=r_t+\gamma(r_{t+1}+\gamma(r_{t+2}+...))=r_t+\gamma_{}R_{t+1}$$

参考：https://juejin.im/post/5d591d6af265da03c8151f4f、https://easyai.tech/ai-definition/reinforcement-learning/

### ☣️ Q-Learning

**Q(s, a)函数(Quality)，质量函数用来表示智能体在s状态下采用a动作并在之后采取最优动作条件下**的打折的未来奖励(先不管未来的动作如何选择)：

$$Q(s_t,a_t)=maxR_{t+1}$$

假设有了这个Q函数，那么我们就能够求得在当前 t 时刻当中，做出各个决策的最大收益值，通过对比这些收益值，就能够**得到 t 时刻某个决策是这些决策当中收益最高。**

$\pi(s)=argmax_aQ(s,a)$

于是乎，根据Q函数的递推公式可以得到：

$$Q(s_t,a_t)=r+\gamma_{}max_aQ(s_{t+1},a_{t+1})$$

**这就是注明的贝尔曼公式。**贝尔曼公式实际非常合理。对于某个状态来讲，最大化未来奖励相当于最大化即刻奖励与下一状态最大未来奖励之和。

**Q-learning的核心思想是：**我们能够通过贝尔曼公式迭代地近似Q-函数。


参考：https://juejin.im/post/5d591d6af265da03c8151f4f

### ☣️ DQN

Deep Q Learning(DQN)是一种融合了神经网络和的Q-Learning方法。

**神经网络的作用**



![img](https://user-gold-cdn.xitu.io/2019/8/18/16ca41b0dc74ff8e?imageView2/0/w/1280/h/960/ignore-error/1)



使用表格来存储每一个状态 state, 和在这个 state 每个行为 action 所拥有的 Q 值. 而当今问题是在太复杂, 状态可以多到比天上的星星还多(比如下围棋). 如果全用表格来存储它们, 恐怕我们的计算机有再大的内存都不够, 而且每次在这么大的表格中搜索对应的状态也是一件很耗时的事. 不过, 在机器学习中, 有一种方法对这种事情很在行, 那就是神经网络.

我们可以将状态和动作当成神经网络的输入, 然后经过神经网络分析后得到动作的 Q 值, 这样我们就没必要在表格中记录 Q 值, 而是直接使用神经网络生成 Q 值.

还有一种形式的是这样, 我们也能只输入状态值, 输出所有的动作值, 然后按照 Q learning 的原则, 直接选择拥有最大值的动作当做下一步要做的动作.

我们可以想象, 神经网络接受外部的信息, 相当于眼睛鼻子耳朵收集信息, 然后通过大脑加工输出每种动作的值, 最后通过强化学习的方式选择动作.

**神经网络计算Q值**

这一部分就跟监督学习的神经网络一样了我，输入状态值，输出为Q值，根据大量的数据去训练神经网络的参数，最终得到Q-Learning的计算模型，这时候我们就可以利用这个模型来进行强化学习了。



![img](https://user-gold-cdn.xitu.io/2019/8/18/16ca41b1338135c6?imageView2/0/w/1280/h/960/ignore-error/1)


参考：https://juejin.im/post/5d591d6af265da03c8151f4f

## 推荐系统

### ✅ 协同过滤模型

**相似性度量方法** 

1.欧几里得距离：该距离只有当两者特征向量中每个特征都较小时，特征向量间距离才比较小。 

2.皮尔森相关系数：该系数度量两个特征向量之间的线性相关性，二者线性相关程度越强，该度量值越高。该度量适用于不同特征的范围不一样的情况。

**基于用户相似度的推荐** 

为用户A提供推荐，首先选择皮尔森相关系数进行度量，计算与A最相似的N个人，然后计算每部电影在相似度加权平均下的加权平均值，以该分值作为每部电影的推荐值。

**基于物品的协作型过滤** 

首先将原始字典的键和值进行调换，此时键为物品，值为一个字典，字典的键为用户，值为分值。根据此字典运行上述的算法，先求物品之间的相似度。

比如要给用户A提供推荐，A做出评价的物品有1，2，3.未作出评价的物品有4，5，6。计算物品4的分值，需要先计算物品4与物品1，2，3的相似度，以相似度作为权值，对物品1，2，3的评分进行加权平均，以此作为对物品4 评分。

```python
# -*- coding: utf-8 -*-
__author__ = 'Bai Chenjia'
from math import *


class recommendation:
    def __init__(self):
        self.critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                                      'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
                        'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5,
                                         'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
                        'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                             'Superman Returns': 3.5, 'The Night Listener': 4.0},
                        'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5,
                                         'Superman Returns': 4.0, 'You, Me and Dupree': 2.5},
                        'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0,
                                         'Superman Returns': 3.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
                        'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0,
                                          'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
                        'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

    def sim_distance(self, person1, person2, new_critics=0):
        """
        使用欧氏距离计算相似度. 返回相似度是一个浮点数值. 返回值为0代表没有相似性.
        首先计算两个人所看相同电影评分的欧氏距离L. 如两人没有观看相同电影则返回0,否则计算相似度公式为 1/(L+1).
        其中分母使用L+1是为了防止出现分母为0的情况. 欧氏距离越大，相似度越小;欧氏距离越小，相似度越大
        """
        if new_critics != 0:
            new_critics = self.transformPrefs()
            movielist1 = new_critics[person1]
            movielist2 = new_critics[person2]   # dict
            commonlist = []        # list
            for item in movielist1:
                if item in movielist2:
                    commonlist.append(item)
            if len(commonlist) == 0:   # 没有公共元素，相似度为0
                return 0
            # 计算欧氏距离(使用列表生成式)
            distance = sum([pow(movielist1[item] - movielist2[item], 2) for item in commonlist])
            sim = 1 / (sqrt(distance) + 1)
            return sim
        else:
            movielist1 = self.critics[person1]
            movielist2 = self.critics[person2]   # dict
            commonlist = []        # list
            for item in movielist1:
                if item in movielist2:
                    commonlist.append(item)
            if len(commonlist) == 0:   # 没有公共元素，相似度为0
                return 0
            # 计算欧氏距离(使用列表生成式)
            distance = sum([pow(movielist1[item] - movielist2[item], 2) for item in commonlist])
            sim = 1 / (sqrt(distance) + 1)
            return sim

    def sim_pearson(self, person1, person2):
        """
        计算皮尔森相关系数.   度量两个向量之间的线性相关性.
        如果一个人总是给出比另一个人更高的分支，但二者的分值只差基本保持一致，则他们仍然存在很好的相关性.
        :return:两个向量之间的皮尔森相关系数，返回浮点数
        """
        movielist1 = self.critics[person1]
        movielist2 = self.critics[person2]
        commonlist = []
        for item in movielist1:
            if item in movielist2:
                commonlist.append(item)  # key
        if len(commonlist) == 0:  # 没有共同项
            return 0

        sum1 = sum([self.critics[person1][item] for item in commonlist])   # person1评分和
        sum2 = sum([self.critics[person2][item] for item in commonlist])   # person2评分和

        sum1sq = sum([pow(self.critics[person1][item], 2) for item in commonlist])  # person1评分平方和
        sum2sq = sum([pow(self.critics[person2][item], 2) for item in commonlist])  # person2评分平方和

        psum = sum([self.critics[person1][item] * self.critics[person2][item]
                    for item in commonlist])  # 交叉项乘积和

        # calculate perarson
        n = len(commonlist)
        num = (psum - (sum1 * sum2 / n))
        den = sqrt((sum1sq - pow(sum1, 2) / n) * (sum2sq - pow(sum2, 2) / n))
        if den == 0:
            return 0
        r = num / den
        return r

    def topMatches(self, person, n=5, similarity=sim_pearson):
        """
        返回与person最相似的n个人的姓名和相似度. 相似度度量选用 皮尔森相关系数
        :param person: 要计算的人的姓名
        :param n: 与person相似度最高的n个人
        :param similarity: 相似度度量方法
        :return: 返回一个list，list中元组存储的形式为（相关性, 姓名）
        """
        if person not in self.critics:
            print "person input error!"
            return 0
        scores = [(similarity(self, person, item), item) for item in self.critics if item != person]
        # print scores[:]
        return sorted(scores, key=lambda it: -it[0])[0:n]  # 取相似度最高的n位

    def getRecommendations(self, person, similarity=sim_pearson):
        """
        用与person的相似度对其他人对person未看过的影片进行加权平均，对person未看影片进行加权打分
        :param person:
        :param similarity: 相似度度量方法
        :return: 返回对 person 未看影片的推荐评分，按评分排序进行推荐
        """
        moviescore_dict = {}  # 影片--影片对应的评分和
        moviesim_dict = {}  # 影片--影片对应的相似度和
        for other in self.critics:
            sim = similarity(self, other, person)  # 人之间的相似度
            if other == person:
                continue
            if sim < 0:
                continue
            for item in self.critics[other]:
                if item not in self.critics[person]:  # 如果person未对该电影进行评价
                    moviescore_dict.setdefault(item, 0)
                    moviescore_dict[item] += self.critics[other][item] * sim

                    moviesim_dict.setdefault(item, 0)
                    moviesim_dict[item] += sim

        result = [(float(moviescore_dict[item] / moviesim_dict[item]), item)
                  for item in moviescore_dict]
        result = sorted(result, key=lambda e1: -e1[0])
        return result

    def transformPrefs(self):
        """
        将self.critics中的 人--电影 的键和值进行调换，反向构造字典
        :return: 键值调换后的 dict
        """
        result = {}
        for person in self.critics:
            for item in self.critics[person]:
                result.setdefault(item, {})
                result[item][person] = self.critics[person][item]
        return result

    def calculateSimilarItems(self, similarity=sim_distance):
        """
        利用反向构造的 物品--人  字典，构造一个物品相似度字典
        字典的键是物品，值是与该物品最相似的n个物品以及物品之间的相似度
        :param n: 与物品最相似的n个物品
        :return: dict
        """
        result = {}
        reverse_critics = self.transformPrefs()
        for movie in reverse_critics:
            scores = [(similarity(self, movie, item, 100), item)
                      for item in reverse_critics if item != movie]
            scores = sorted(scores, key=lambda it: -it[0])       # 取相似度最高的n位
            result[movie] = scores
        return result

    def getRecommendedItems(self, user):
        """
        指定一个名字user，计算对该用户的电影推荐
        首先利用 calculateSimilarItems方法计算物品之间的相似度dict
        根据该用户对曾经看过的影片的评分以及该电影与未看过电影的相似度估计用户对未看电影的评分，根据评分进行推荐
        :param user: 指定的人
        :return:
        """
        userRatings = self.critics[user]  # 返回该用户所看电影和评分
        itemMatch = self.calculateSimilarItems()  # 返回物品的相似度字典
        scores = {}
        totalsim = {}
        for item, rating in userRatings.items():
            for similarity, item2 in itemMatch[item]:
                if item2 in userRatings:   # 若该电影被用户评分
                    continue
                scores.setdefault(item2, 0)
                scores[item2] += similarity * rating
                totalsim.setdefault(item2, 0)
                totalsim[item2] += similarity
        ranking = [(score / totalsim[item], item) for item, score in scores.items()]
        ranking = sorted(ranking, key=lambda it: -it[0])
        return ranking


if __name__ == '__main__':
    system = recommendation()

    # 1.欧几里得距离计算相似度
    print "\n欧几里得相似度:", system.sim_distance('Lisa Rose', 'Gene Seymour')

    # 2.皮尔森相关系数计算
    print "\n皮尔森相关系数:", system.sim_pearson('Lisa Rose', 'Gene Seymour')

    # 3.计算与某人相似度最高的n个人
    print "\n与 Toby 相似度最高的n个人", system.topMatches('Toby', n=3)[:]

    # 4.输入人名，对该人未观看的影片推荐
    print "\n给 Toby 未观看影片的推荐是", system.getRecommendations('Toby')[:]

    # 5.构建物品相似度字典
    print "\n电影之间的相似度字典是："
    for key, value in system.calculateSimilarItems().iteritems():
        print key, ":", value[:]

    # 6.根据物品相似度，输入人名，推荐影片
    print "\n根据电影相似度给 Toby 推荐的影片是：", system.getRecommendedItems('Toby')[:]
```

参考资料：https://blog.csdn.net/bcj296050240/article/details/50810662

## 回归模型

### ☣️ 线性规划（Linear Regression）



### 逻辑回归（Logistic Regression）

![img](https://yuanxiaosc.github.io/2018/06/21/%E6%94%B9%E8%BF%9B%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E2%80%94%E2%80%94%E4%BA%A4%E5%8F%89%E7%86%B5/one_1.png)



### SoftMax

<img src="https://pic1.zhimg.com/v2-11758fbc2fc5bbbc60106926625b3a4f_1200x500.jpg" style="zoom:70%">

```python
scores = np.array([123, 456, 789])    # example with 3 classes and each having large scores
scores -= np.max(scores)    # scores becomes [-666, -333, 0]
p = np.exp(scores) / np.sum(np.exp(scores))
```

主要用于多分类回归，可以对向量进行处理，最后得出的概率总和是1

参考资料：https://zhuanlan.zhihu.com/p/25723112（softmax应用）

### ☣️ 似然函数（Likelihood Function）



### ☣️ 贝叶斯网络（Bayes Network）





参考资料：https://cloud.tencent.com/developer/article/1102103（分类算法总结）

## 统计模型-蒙圈

### ☣️ 条件随机场（Conditional Random Field）



### ✅ 马尔可夫模型（Markov Models）

**概述**：马尔可夫过程中，任何一个状态，只与前一个状态相关

<img src="https://pic4.zhimg.com/80/648a55725e67d718d97d6a475891d70b_hd.jpg" style="width:50%"/>

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

### 💤 隐马尔可夫模型（Hidden Markov Models）

#### ✅ 前言

隐马尔可夫模型即马尔可夫链上加了一层随机过程，一般采用Baum-Welch算法和Viterbi算法进行求解

![preview](https://pic2.zhimg.com/792e033ff9b0418b3b6c9bbaef30fd83_r.jpg)

#### 🆘 Baum-Welch算法

一种EM算法

#### 🆘 Viterbi算法



#### 🆘 源码-Baum-Welch算法

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

### 💤 蒙特卡罗方法（**Monte Carlo** Method）

#### ✅ 基础模型

**概述**：通过大量随机样本，去了解一个系统，进而得到所要计算的值。它非常强大和灵活，又相当简单易懂，很容易实现。对于许多问题来说，它往往是最简单的计算方法，有时甚至是唯一可行的方法。

**举例**：在区域内随机产生10000个点，落在圆内的点与圆外的点比例计划，可以推导𝝅的近似值

<img src="http://www.ruanyifeng.com/blogimg/asset/2015/bg2015072604.jpg" style="width:30%"/>

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

#### 🆘 Inverse CDF 方法

**原理**：经典且常见的模型如指数分布、𝛾 分布、t 分布、F 分布、β 分布、Dirichlet 分布都是有的，可以方便采样，但是对于相对复杂的分布，就需要设计采样策略，比如Inverse CDF（Cumulative Distribution Function）方法，CDF可以由概率密度函数（PDF，Probability Density Function）进行积分得到

editing.....

#### 🆘 拒绝接受采样

**案例**：假设使用 ![[公式]](https://www.zhihu.com/equation?tex=U%280%2C1%29) 来作为“proposal distribution” ![[公式]](https://www.zhihu.com/equation?tex=G) ，这样 ![[公式]](https://www.zhihu.com/equation?tex=g%28x%29%3D1%5Cforall+x%5Cin+%5B0%2C1%5D) 。如下图所示，我们每次生成的两个样本 ![[公式]](https://www.zhihu.com/equation?tex=Y) 与 ![[公式]](https://www.zhihu.com/equation?tex=U) ，对应下图中矩形内的一点 ![[公式]](https://www.zhihu.com/equation?tex=P%28Y%2CU%E2%88%97c%E2%88%97g%28Y%29%29) 。接受条件 ![[公式]](https://www.zhihu.com/equation?tex=U%5Cleqslant+f%28Y%29c%E2%88%97g%28Y%29) ，即 ![[公式]](https://www.zhihu.com/equation?tex=U%E2%88%97c%E2%88%97g%28Y%29%5Cleqslant+f%28Y%29) 的几何意义是点 ![[公式]](https://www.zhihu.com/equation?tex=P) 在 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 下方，不接受 ![[公式]](https://www.zhihu.com/equation?tex=Y) 的几何意义是点 ![[公式]](https://www.zhihu.com/equation?tex=P) 在 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 的上方。在 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 下方的点(o形状)满足接受条件，上方的点(+形状)不满足接受条件。 

<img src="https://pic1.zhimg.com/80/v2-ba61bc580f87754f4e35675c6ac1b23c_hd.jpg" style="width:50%"/>

**代码实现**：假如我们的目标概率密度函数是 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29%3D%5Cfrac%7B%28x-0.4%29%5E%7B4%7D%7D%7B%5Cint_%7B0%7D%5E%7B1%7D%28x-0.4%29%5E%7B4%7Ddx%7D) ，对此分布生成样本。

```python
import random
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%matplotlib inline
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

def AceeptReject(split_val):
    global c
    global power
    while True:
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        if y*c <= math.pow(x - split_val, power):
            return x

power = 4
t = 0.4  
sum_ = (math.pow(1-t, power + 1) - math.pow(-t, power + 1)) / (power + 1)  #求积分
x = np.linspace(0, 1, 100)
#常数值c
c = 0.6**4/sum_
cc = [c for xi in x]
plt.plot(x, cc, '--',label='c*f(x)')
#目标概率密度函数的值f(x)
y = [math.pow(xi - t, power)/sum_ for xi in x]
plt.plot(x, y,label='f(x)')
#采样10000个点
samples = []
for  i in range(10000):
    samples.append(AceeptReject(t))
plt.hist(samples, bins=50, normed=True,label='sampling')
plt.legend()
plt.show()
```

<img src="https://pic2.zhimg.com/80/v2-8ca3019c1c1b0a878f44f51458c5b15d_hd.jpg" alt="img" style="zoom:70%;" />

#### 🆘 自适应的拒绝采样（Adaptive Rejection Sampling）

editing.....

参考资料：http://www.ruanyifeng.com/blog/2015/07/monte-carlo-method.html 蒙特卡洛、https://blog.csdn.net/baimafujinji/article/details/51407703 蒙特卡洛采样之拒绝采样（Reject Sampling）

### 💤 马尔可夫链蒙特卡洛（Markov chain Monte Carlo）

#### ✅ 背景

**动机一**

假如你需要对一维随机变量$X$进行采样， ![[公式]](https://www.zhihu.com/equation?tex=X) 的样本空间是$\{1,2,3\}$ ，且概率分别是 $\{1/2,1/4,1/4\}$ ，这很简单，只需写这样简单的程序：首先根据各离散取值的概率大小对 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D)区间进行等比例划分，如划分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C0.5%5D%2C%5B0%2C5%2C0.75%5D%2C%5B0.75%2C1%5D) 这三个区间，再通过计算机产生[0,1]之间的伪随机数，根据伪随机数的落点即可完成一次采样。接下来，假如 ![[公式]](https://www.zhihu.com/equation?tex=X) 是连续分布的呢，概率密度是$f(X)$，那该如何进行采样呢？聪明的你肯定会想到累积分布函数， $P(X<t)=\int_{-\infty}^tf(x)dx$ ，即在  间随机生[0,1]成一个数 a ，然后求使得使$P(s<t)=a$成立的 ![[公式]](https://www.zhihu.com/equation?tex=t) ， ![[公式]](https://www.zhihu.com/equation?tex=t) 即可以视作从该分部中得到的一个采样结果。这里有两个前提：一是概率密度函数可积；第二个是累积分布函数有反函数。假如条件不成立怎么办呢？MCMC就登场了。

**动机二**

假如对于高维随机变量，比如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BR%7D%5E%7B50%7D) ，若每一维取100个点，则总共要取 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B100%7D) ，而已知宇宙的基本粒子大约有 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B87%7D) 个，对连续的也同样如此。因此MCMC可以解决“维数灾难”问题。

#### 🆘 M-H采样



#### 🆘 Gibbs采样



参考资料：https://zhuanlan.zhihu.com/p/37121528 MCMC——其实没太看懂
