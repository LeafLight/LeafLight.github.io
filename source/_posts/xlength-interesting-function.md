---
title: xlength-an interesting function
date: 2022-04-10 14:26:26
tags:["Python", "functools", "lambdaFunction"]
---

## Background

When reading the source code of [DLEPS](https://github.com/kekegg/DLEPS/bolb/main/code/Preprocess/preprocess.ipynb), some interesting codes caught my eyes, including: `six.string_types`, `get_zinc_tokenizer()`, `xlength()`, and so on.
## Main

### Pre-knowledge

#### lambda

A lambda function in _python_ is in some ways like a simple `def` fucntion with only a `return` line.

```python
# use `def` function to represent `lambda x: x * x`
def f(x):
	return x * x
```

#### functools.reduce()

`functools` is a python module which contains some advanced functions. The `functools.reduce()` will apply a given function to a iterable sequence and a initial value is optional.

```python
# use `functools.reduce()` to realize `sum()`
# functools.reduce(func, iter[, init])
from functools import reduce
y = [1, 2, 3]
sum = reduce(lambda x1, x2: x1 + x2, y, 0)
print(sum)
# 6
```
In the case above, the first step of work by `reduce` is to apply the given function with `init` 0  and the first element of the `iter` y as input(so the given function must receive 2 parameters) to get the result 1. Then apply the function with the result 1 and the second element 2, and repeat it until the last one of the list.

If no `init` was given, it would first apply the given function with the first two elements of the `iter` y.

### xlength()

The source code of xlength:
```python
def xlength(y):
	return reduce(lambda sum, element: sum + 1, y, 0)
```

It really confused me when I first ran this code and found that it returns the length of the list. Then I realized it that it works by the interesting applying of `init` and `func`. The `func` desert the `element`, which means it has no effect on the results the `func` returns. So the `sum` is in some ways a counter to count the length of the `iter` by `sum + 1` 
