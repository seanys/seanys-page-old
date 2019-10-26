---
layout: post
title: "Python 数据处理"
subtitle: 'Machine Learning Part 1'
author: "sean"
header-img: "img/post-bg-ai.jpeg"
tags:
  - Python
  - 数据挖掘
---



## Pandas

### CSV处理

```python
import pandas
df = pandas.read_csv('hrdata.csv')
print(df) #读取CSV，但是输出的是全部

df2 = pandas.read_csv('hrdata.csv', index_col='Name') # 读取特定的一列
print(df2)

df3 = pandas.read_csv('hrdata.csv', index_col='Name', parse_dates=['Hire Date']) # parse_dates将该列数据强制处理成日期
print(df3)

df4 = pandas.read_csv('hrdata.csv', 
            index_col='Employee', 
            parse_dates=['Hired'], 
            header=0,  # 忽略现有的名称
            names=['Employee', 'Hired','Salary', 'Sick Days'])  # 采用新的列名
print(df4)

df5 = pandas.read_csv('hrdata.csv', 
            index_col='Employee', 
            parse_dates=['Hired'],
            header=0, 
            names=['Employee', 'Hired', 'Salary', 'Sick Days'])
df5.to_csv('hrdata_modified.csv') # 重新写入新的csv

print(df5.head(5)) # 前五行
print(df5.tail(5)) # 后五行

print(df5['Salary'][1]) # 输出某一列某个位置的值

df = pd.read_csv('file.csv',header=0,index_col=0) # 读取某一列
df = pd.read_csv('data.csv',nrows =5) # 读取某一行

for index, row in df.iterrows(): # 遍历行处理
    print row["c1"], row["c2"]

```



## Open CV

### 绘图

1.画线

需要告诉函数这条线的起点和终点。

```python
import numpy as np
import cv2

#Create a black image
img = np.zeros((512,512,3),np.uint8)

#draw a diagonal blue line with thickness of 5 px
cv2.line(img,(0,0),(260,260),(255,0,0),5)

#为了演示，建窗口显示出来
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)#定义frame的大小
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

2.画矩形

需要告诉函数左上角顶点和右下角顶点的坐标

```python
import numpy as np
import cv2

#Create a black image
img = np.zeros((512,512,3),np.uint8)

cv2.rectangle(img,(350,0),(500,128),(0,255,0),3)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)#定义frame的大小
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

3.画圆

需要指定圆心坐标和半径大小，可以在上面矩形中画个圆

```python
import numpy as np
import cv2

#Create a black image
img = np.zeros((512,512,3),np.uint8)

cv2.rectangle(img,(350,0),(500,128),(0,255,0),3)#矩形
cv2.circle(img,(425,63),63,(0,0,255),-1)#圆，-1为向内填充

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)#定义frame的大小
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

4.画椭圆

一个参数是中心点的位置坐标。 下一个参数是和短的度。椭圆沿时方向旋的度。椭圆弧演 时方向始的度和结束度如果是 0 很 360就是整个椭圆。查看 cv2.ellipse() 可以得到更多信息。

```python
import numpy as np
import cv2

#Create a black image
img = np.zeros((512,512,3),np.uint8)

cv2.ellipse(img,(256,256),(100,50),0,0,360,255,-1)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)#定义frame的大小
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

5.画多边形

需要指定每个顶点的坐标，构建一个大小相等于行数X1X2的数组，行数就是点的数目，这个数组必须为int32。

```python
import numpy as np
import cv2

#Create a black image
img = np.zeros((512,512,3),np.uint8)

pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts = pts.reshape((-1,1,2))
#这里reshape的第一个参数为-1，表明这一维度的长度是根据后面的维度计算出来的
cv2.polylines(img,[pts],True,(0,255,255)) 
#注意第三个参数若是False，我们得到的是不闭合的线

#为了演示，建窗口显示出来
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',1000,1000)#定义frame的大小
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

cv2.polylines() 可以用来画很多条线。只把想画的线放在一 个列中将个列传给函数就可以了。每条线会独立绘制。会比用 cv2.line() 一条一条的绘制快一些。

7.显示结果可以用以下代码

```python
winname='example'
cv2.namedWindow(winname)
cv2.imshow(winname,img)
cv2.waitKey(0)
cv2.destroyAllWindow(winname)
```

## Matplotlib







