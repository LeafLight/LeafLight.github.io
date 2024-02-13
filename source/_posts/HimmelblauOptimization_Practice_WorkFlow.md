---
title: Himmelblau Function -- Optimization Practice
tags: ["Himmelblau Function", "Optimization", "Practice"]
category:
- [MachineLearning]
---

## What is Himmelbau Function?

It is a function of two variables. It has a bowl-like 3d shape and four points which all have a minimum value of zero. This is a function wildly used to examine a optimization algorithm.

It can be defined in python like this.
```python
def himmelblau(x):
	return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
```

The visualization of it can be realized by code below.
```python
import numpy as np
import matplotlib.pyplot as plt
def himmelblau(x):
	return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y range:', x.shape, y.shape)
X, Y = np.meshgrid(x,y)
print('X, Y maps:', X.shape, Y.shape)

Z = himmelblau([X,Y])

fig = plt.figure('himmelblau')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_unit(60, -30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show
```

## The purpose of this practice
* Use a Gradient Descent to find the minimum of the function.

* Have a deeper understanding of the truth that the initialization of a GD model is important.

## Source Code

(in x/code/py)

Learning from [bilibili: the best pytorch tutorial of 2021](https://www.bilibili.com/video/BV1US4y1M7fg?p=46)
This script didn't use auto _optimizer_, which aims to use less module-source feature.It causes some problems, which help me understand how _pytorch_ works.
```python
#!/bin/python3
#-*-coding:UTF-8-*-
#Author:LeafLight
#Date: 2022-02-18
import numpy as np
import matplotlib.pyplot as plt
import torch
def himmelblau(x):
    return (x[0] ** 2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

########################################
#visualization,which is not required in this script
##x = np.arange(-6, 6, 0.1)
##y = np.arange(-6, 6, 0.1)
##print('x, y range:', x.shape, y.shape)
##X, Y = np.meshgrid(x,y)
##print('X, Y maps:', X.shape, Y.shape)

##Z = himmelblau([X,Y])

# show the visualization of Himmelblau function 
##fig = plt.figure('himmelblau')
##ax = fig.gca(projection='3d')
##ax.plot_surface(X, Y, Z)
##ax.view_init(60, -30)
##ax.set_xlabel('x')
##ax.set_ylabel('y')
##plt.show()
#end of the visualization
########################################

# Input: A pair of initialization of x,y of Himmelblau function
point = [0, 0]
point[0] = float(input("a list of initialization list for the model(x):"))
point[1] = float(input("a list of initialization list for the model(y):"))
t_point = torch.tensor(point, requires_grad=True)
# learning rate 
lr = 1e-4
# A loop of 2e4 epoches
for epoch in range(20000):
    # loop: calculate the function's value   
    value = (t_point[0] ** 2 + t_point[1] -11) ** 2 + (t_point[0] + t_point[1] ** 2 - 7) ** 2
    # loop: calculate the grad
    value.backward()
    if epoch % 2e2 == 0:
        # loop: show the value
        print("epoch: ", epoch, ", x: ", t_point[0], ", y:", t_point[1], ", value: ", value.data) 
    if value != 0: 
        # loop: update the x,y
        t_point.data[0] = t_point.data[0] - t_point.grad[0] * lr
        t_point.data[1] = t_point.data[1] - t_point.grad[1] * lr
        # reset the grad.
        t_point.grad.zero_()
    else:
        break
print("____________________")
print("final: ",value)
print("x: ",t_point[0].data)
print("y: ",t_point[1].data)
```
## Some problems

1. Take care of overflow error, especially when using high power.(Maximum of int64tensor: 2 ^ 63 - 1)
2. When changing the value of a tensor that requires grad. ,assign the value to `tensor1.data` instead of to `tensor1` directly, which will cause the __Error__, `Leaf variable was used in a inplace operation`
