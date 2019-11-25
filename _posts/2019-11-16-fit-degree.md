---

layout: post
title: "Get fit degree in python"
subtitle: 'Combination evalution method for two pieces'
author: "sean"
header-img: "img/post-bg-os-metro.jpg"
tags:
  - Irregular bin packing
---

## Introducation

If we want to evaluate the combiantion results, we need use some evaluation criterion. Dominate 







## What is fit degree

Fit degree can be showed by the extended polygon (I can't find the paper this method proposed from)

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g94xuars3kj313c0i644h.jpg" alt="image-20191120234722371" style="zoom:30%;" />

## How to get the fit degree of one point

### **Expansile** polygon

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g98yummpqoj31b00t8gnf.jpg" alt="image-20191124112553676" style="zoom:40%;" />

```python
    '''
        求解凸多边形的近似多边形，凹多边形内凹部分额外处理
    '''
    def similarPoly(self,poly):
        change_len=25
        extend_poly=poly+poly
        Poly=Polygon(poly)
        new_edges=[]
        # 计算直线平移
        for i in range(len(poly)):
            line=[extend_poly[i],extend_poly[i+1]]
            new_line=self.slideOutLine(line,Poly,change_len)
            new_edges.append(new_line)
        
        # 计算直线延长线
        new_poly=[]
        new_edges.append(new_edges[0])
        for i in range(len(new_edges)-1):
            new_poly.append(self.extendInter(new_edges[i],new_edges[i+1]))

        return new_poly
    
    '''
        向外平移直线
    '''
    def slideOutLine(self,line,Poly,change_len):
        pt0=line[0]
        pt1=line[1]
        mid=[(pt0[0]+pt1[0])/2,(pt0[1]+pt1[1])/2]
        if pt0[1]!=pt1[1]:
            k=-(pt0[0]-pt1[0])/(pt0[1]-pt1[1]) # 垂直直线情况
            theta=math.atan(k)
            delta_x=2*math.cos(theta)
            delta_y=2*math.sin(theta)
            if Poly.contains(Point([mid[0]+delta_x,mid[1]+delta_y])):
                delta_x=-delta_x
                delta_y=-delta_y
            return [[pt0[0]+change_len*delta_x,pt0[1]+change_len*delta_y],[pt1[0]+change_len*delta_x,pt1[1]+change_len*delta_y]]
        else:
            delta_y=1
            if Poly.contains(Point([mid[0],mid[1]+delta_y])):
                delta_y=-delta_y
            return [[pt0[0],pt0[1]+change_len*delta_y],[pt1[0],pt1[1]+change_len*delta_y]]
    '''
    		延长直线
    '''
    def extendLine(self,line):
        pt0=line[0]
        pt1=line[1]
        vect01=[pt1[0]-pt0[0],pt1[1]-pt0[1]]
        vect10=[-vect01[0],-vect01[1]]
        new_pt1=[pt0[0]+vect01[0]*3,pt0[1]+vect01[1]*3]
        new_pt0=[pt1[0]+vect10[0]*3,pt1[1]+vect10[1]*3]
        return [new_pt0,new_pt1]

    '''
        获得延长线的交点
    '''
    def extendInter(self,line1,line2):
        line1_extend=self.extendLine(line1)
        line2_extend=self.extendLine(line2)
        inter=mapping(LineString(line1_extend).intersection(LineString(line2_extend)))
        return inter["coordinates"] 
```

### Area of Polygons' intersection





## How to find the fittest point

<img src="https://tva1.sinaimg.cn/large/006y8mN6gy1g98z3jddcij31h60u0gre.jpg" alt="image-20191124113427556" style="zoom:30%;" />

When we slide the adjoin polygon keep adjoin's reference point in no-fit polygon, the intersection's area of these two expansile polygon's will change. Every edge of nfp will have the fittest position for reference point.

The change function of fit degree in an edge is a convex polygon.(I don't know how to express this)























