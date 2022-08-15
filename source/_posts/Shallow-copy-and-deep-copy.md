---
title: Shallow copy and deep copy in Python
tags: ["Python", "Copy"]
date: 2022-08-16 02:38:53
---
## Reference
[geeksforgeeks](https://www.geeksforgeeks.org/copy-python-deep-copy-shallow-copy/)
[numpy.copy](https://numpy.org/doc/stable/reference/generated/numpy.copy.html)
## Deep copy
Create a new object and them population it with the copies of the childs of the original object recursively.
In short, any changes made to the copy __will not reflect__ in the original object.
## Shallow copy
Just create a new variable that shares the same reference to the original object.
It means that any changes made will effect both the copy and the original variable.

## Why it is important?
It once confused me why `.copy()` is used instead of simple assignment to do copy in some scripts. Now just look at the example below.
```python
def test_copy(l):
	l[0] = 'changed'
	return l
a = [1, 2, 3 ,4]
b = test_copy(a)
print(a == b)
# It will print `True`
# It means that we do a shallow copy when simply use assignment and mistakes tends to happen when we don't pay enough attention to it.
```

## How to do deep copy?
1. Use the `copy` moduel.
```python
import copy
l1 = [1, 2, 3, 4]
l2 = copy.deepcopy(l1)
```

2. Use `copy` in NumPy.
```python
x = np.array([1, 2, 3])
y = x
z = np.copy(x)
# Note that the doc call y a reference to x and z the copy of x.
```

3. Use `DataFrame.copy` in pandas
```python
x = pd.Series([1, 2], index=["a", "b"])
shallow = x.copy(deep=False)
deep = x.copy()
# shallow is x: False
# shallow.values is x.values and shallow.index is x.index: True
# deep is x: False
# deep.values is x.values and deep.index is x.index: False
```

4. Use `clone` in Pytorch
```python
t = torch,rand(2,10)
t_copy = t.clone()
# it will be recorded by `autograd` because it is a pytorch operation
# When it comes to Module, there is no clone method available so just use `copy.deepcopy()` instead.
```

