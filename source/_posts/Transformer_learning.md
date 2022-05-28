---
title: A deeper insight into the training of Transformer Autoencoder
tags: ["Transformer", "MachineLearning", "Pytorch"]
---

## Transformer Abstract

_Transformer_ is a well-known neural network architecture, I have learned about this before from Mu Li's [video on bilibili](https://www.bilibili.com/video/BV1pu411o7BE). The video is brief but cover the details about the attention mechanism the _Transformer_ uses. 

However, When I want to practice using Pytorch code following another [video](https://m.bilibili.com/video/BV19Y411b7qx), the training way of Transformer is a little different with normal CNN. It is like RNN or LSTM, but I am not very familiar with NLP and associated architectures.

## Transformer training

In the practice video above, what confused me is _p9_(the 9th video). 

Q1: The video's task is translation between language _x_ and _y_. But the shape of _x sentence_ is `[b, 50, e]` while that of _y sentence_ is `[b, 51, e]`. 

Q2: Then when training the model `y[:, :-1, :]` was used as target, while `y[:, 1:, :]` was used to do loss calculation.

Q3: When using the model to do `pred(x)`(which returns y, referred as `out` to avoid mistaking), the `out` was generated one word by one word.

## Understanding

A1: It's for the convenience of slicing. The reason for slicing is to do things about Q2.

A2: You may notice that by the special way of slicing, the target and loss label were one-position crossing. With the help of `tril_mask`, the model is able to learn to predict the next word given the preceding word.

A3: Since its ability is to predict the next one word, the model has to do prediction in this way.
