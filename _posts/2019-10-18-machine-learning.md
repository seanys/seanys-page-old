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

备注：部分算法在多个分类中均有应用，可能只写在一个分类中，或者放在了回归模型中，网页版不支持公式语句，采用图片模式


## 机器学习与数据挖掘

### ✅ 决策树（Decision Tree）

#### ✅ 前言

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

#### ✅ ID3算法

熵这个概念最早起源于物理学，在物理学中是用来度量一个热力学系统的无序程度，而在信息学里面，熵是对不确定性的度量。在1948年，香农引入了信息熵（information entropy），将其定义为离散随机事件出现的概率，一个系统越是有序，信息熵就越低，反之一个系统越是混乱，它的信息熵就越高。所以信息熵可以被认为是系统有序化程度的一个度量。

**案例**：对于有K个类别的分类问题来说，假定样本集合D中第 k 类样本所占的比例为pk（k=1,2,...,K）,则样本集合D的信息熵定义为: 

<img src="https://tva1.sinaimg.cn/large/006y8mN6ly1g83aclz7lhj30d602it8q.jpg" alt="image-20191019100957943" style="zoom:50%;" />

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

#### ✅ C4.5算法

**背景**：我们知道信息增益会偏向取值较多的特征，使用信息增益比可以对这一问题进行校正。

**信息增益**：Gain(D,A) = H(D) – H(D|A)，用于ID3的判断

**信息增益比**：特征A对训练数据集D的信息增益比GainRatio(D,A)定义为其信息增益Gain(D,A)与训练数据集D的经验熵H(D)之比：

<img src="https://shuwoom.com/wp-content/uploads/2018/10/aae983046e32f250faf6daaaf376ec90.png" alt="img" style="zoom:50%;" />

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

#### ✅ CART算法

**概述**：CART分类树预测分类离散型数据，采用基尼指数选择最优特征，同时决定该特征的最优二值切分点。

**概率分布的基尼指数定义**：假设有K个类，样本点属于第k个类的概率为Pk

![img](https://pic3.zhimg.com/80/v2-780d955260d9a2ba8508c1601588b88a_hd.jpg)

**集合D的基尼指数**：

![img](https://pic2.zhimg.com/80/v2-95300197189b4b1b65eb42d1a6bbd7fd_hd.jpg)

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



## 学习网络

### ☣️ 梯度下降



### ☣️ BP网络



### ☣️ RNN系列

#### ☣️ RNN模型



![roolled RNN](https://caicai.science/images/attention/roolled%20rnn.png)

#### ☣️ LSTM



#### ☣️ GRU



## 相关内容

### ☣️ 梯度爆炸问题





### ☣️ 代价函数

#### ✅ 二次代价函数

![img](https://img-blog.csdn.net/20160402180717102)

![img](https://img-blog.csdn.net/20160402175137034)

#### ✅ 交叉熵损失函数

![img](https://img-blog.csdn.net/20160402172100739)

参考资料：https://blog.csdn.net/u014313009/article/details/51043064



## 前沿研究

### ✅ Sequence to Sequence

![这里写图片描述](https://img-blog.csdn.net/20171101103708217)

![img](https://images2018.cnblogs.com/blog/1192699/201808/1192699-20180806141638641-1078830067.png)

**概述**：一种函数映射模型，具有弱约束的多对多对应关系。

**原理**：左侧Encoder编码将输入序列转化成一个固定长度的向量编码，右侧Decoder解码将之前生成的固定向量再转化成输出序列，编解码部分可以采用CNN、RNN、LSTM、GRU、BLSTM等实现。可以预测任意的序列对应关系。

**意义**：第一次提出了Encoder-Decoder模型，主要用于序列预测

**问题**：Encoder是将输入内容压缩到一个固定长度的向量中，语义编码是固定长度，然后计算出内容进行Decoder

### ✅ Attention Mechanism

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



### ✅ 逻辑回归（Logistic Regression）

![img](https://yuanxiaosc.github.io/2018/06/21/%E6%94%B9%E8%BF%9B%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95%E2%80%94%E2%80%94%E4%BA%A4%E5%8F%89%E7%86%B5/one_1.png)



### ✅ SoftMax

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

假如你需要对一维随机变量$X$进行采样， ![[公式]](https://www.zhihu.com/equation?tex=X) 的样本空间是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2C2%2C3%5C%7D) ，且概率分别是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2F2%2C1%2F4%2C1%2F4%5C%7D) ，这很简单，只需写这样简单的程序：首先根据各离散取值的概率大小对 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D)区间进行等比例划分，如划分为 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C0.5%5D%2C%5B0%2C5%2C0.75%5D%2C%5B0.75%2C1%5D) 这三个区间，再通过计算机产生 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 之间的伪随机数，根据伪随机数的落点即可完成一次采样。接下来，假如 ![[公式]](https://www.zhihu.com/equation?tex=X) 是连续分布的呢，概率密度是 ![[公式]](https://www.zhihu.com/equation?tex=f%28X%29) ，那该如何进行采样呢？聪明的你肯定会想到累积分布函数， ![[公式]](https://www.zhihu.com/equation?tex=P%28X%3Ct%29%3D%5Cint+_%7B-%5Cinfty%7D%5E%7Bt%7Df%28x%29dx) ，即在 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C1%5D) 间随机生成一个数 ![[公式]](https://www.zhihu.com/equation?tex=a) ，然后求使得使 ![[公式]](https://www.zhihu.com/equation?tex=P%28x%3Ct%29%3Da) 成立的 ![[公式]](https://www.zhihu.com/equation?tex=t) ， ![[公式]](https://www.zhihu.com/equation?tex=t) 即可以视作从该分部中得到的一个采样结果。这里有两个前提：一是概率密度函数可积；第二个是累积分布函数有反函数。假如条件不成立怎么办呢？MCMC就登场了。

**动机二**

假如对于高维随机变量，比如 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb%7BR%7D%5E%7B50%7D) ，若每一维取100个点，则总共要取 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B100%7D) ，而已知宇宙的基本粒子大约有 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B87%7D) 个，对连续的也同样如此。因此MCMC可以解决“维数灾难”问题。

#### 🆘 M-H采样



#### 🆘 Gibbs采样



参考资料：https://zhuanlan.zhihu.com/p/37121528 MCMC——其实没太看懂
