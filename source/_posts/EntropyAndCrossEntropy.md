---
title: An interesting understanding of Entropy and Cross Entropy
tags: ["Entropy", "machine learning", "Note", "Information theory"]
---

## What is Entropy? What is Information Entropy? What is the difference between them?
When learning the [Pytorch turorial from Bilibili](https://www.bilibili.com/video/BV1US4t1M7g?p=48), the appearance of entropy and its abstract definition really confused me.

Here is the answer for entropy from Encyclopedia Britannica:
> Entropy, the measure of a system's thermal energy per unit temperature that is unavailable for doing useful work.

Here is the answer for infromation entropy from wiki:
> In information theory, the entropy of a random variable is the average level of "information", "surprise", or "uncertainty" inherent to the variable's possible outcomes.

These definitions above may do not help at all. And the simple answer of the difference between them is that they are just the same thing in different fields.

## Why do we use Information Entropy?

When faced with classification problems(or sometimes logistic regression), using Information Entropy instead of final accuracy as loss is important. (though it seems like we use entropy because of the weakness of using final accuracy instead of the strength of information entropy,2022/2/21)

That is because the output of a classification model are usually a list of transformed(or cutted) probabilities(like p>0.5?t=1:t=0), which means using the final accuracy will lead to some problems,such as:

* the accuracy remains unchanged when the Weights of a net work are changed.(e.g., p changed from 0.3 to 0.4, but it doesn't help)
* the gradient is not continuous since the accuracy is not continuous.

(Here I wondered that why not use MSE of p and 0 or 1 as loss, and then I learned that it does work(Actually, this method is used in [the MNIST test before](https://leaflight.github.io/2022/02/17/ClassificationAndMNIST/)). No one can tell which one is better than another. But it is an interesting way to understand information, and a useful way to evaluate the loss, so just go on.)

## how to understand Information Entropy in a easy way?

To understand a math definition, usually the combination of its actual math fomula and a scene leading-in will help.

### The math formula of Information Entropy

H(p) = -sum(p.i * log(p.i))

note:
* p: p.1, p.2, ..., p.n
* the base of log can be any number when comparing the H() of different samples.

### Scene leading-in
Reference: [Bilibili](https://www.bilibili.com/video/BV1Ga41127Zu)
Let's imagine a scene that a dice is thrown and we don't know the number of the up face. Here are 3 pieces of information:

1. The number is larger than 0
2. The number is larger than 3
3. The number is larger than 5

It is obvious that the third information is most valuable.So it is possible to compare the value of different information. But we want to evaluate the value of information by quantity. So let's imagine another scene of a ball-number-guessing game. In this game, there are _n_ balls in a box,which all have a number on it surface from 1 to n. One of them will picked out, and we need to guess the number on it, or we can to pay 1 dollar for asking for a question about whether the it is larger than a certain number. We all know dichotomy is the best way if we are willing to pay for the answer:

1. When there 2 balls,we need to ask 1 time,because 2<2^2
2. When there 4 balls,we need to ask 2 times,because 4<2^3
3. When there 8 balls ,we need to ask 3 times,because 8<2^4

So if we can see the number of the ball directly, we can:
1. save 1 dollar
2. save 2 dollars
4. save 3 dollars

Then we learned that the value of the information about what the number of the ball is depends on the probability that we can guess it. In another way, the information's value depends on probability we get the accurate answer without the information.

And we can be more mathematical, in this scene, the value of the information can be calculated by the formula below:
`value = -log2(p)`
note: "2" here is the information unit

We can find that the more uncertainty the information can clear, the more value the information has.

But we what we need to understand is informatiom entropy, so let's imagine anthor scene:
It is still a game about balls, but this time there are only a white ball and a black ball in the box.And _Pw_ is the probability of picking out the white ball,while _Pb_ is for the black one. We guess the ball is black. Fortunately, your friend saw the ball's color, and he said:
1. The ball is black
2. The ball is white

This friend's sight is good.So in which condition, he helps us more?
As we learned before, we can evaluate the information's value by quantity. So:
1. -log2(Pb)
2. -log2(Pw)
note:"2" here is not very important because it doesn't matter when we just want to compare two value

This comparison is in sense because if the _Pb = 0.9_, we can guess it by ourselve more easily, which means the information of this friend seems not very valuable.
Now, what is the average value of information given by this friend?
```
a_value = -sum(p.i*log(p.i))
```

Amazing, it is the formula of Entropy. The more chaos the system is, the more average value of a accurate information has, so it makes sense!

## Cross Entropy

The same as _Information Entropy_, there is a mathematical formula and  scene leading-in for _Cross Entropy_

### Formula

H(p,q) = -sum(p.i * log(q.i))

### Scene leading-in

Here we need to know that Information Entropy can be used to stand the shortest encoding length of a system.For example, To encoing a system of A,B,C,D(their probabilities are 1/2,1/4,1/8,1/8 respectively) Then the shortest average encoding length of this system is H = 1/2 * 1 + 1/4 * 2 + 1/8 * 3 + 1/8 * 3 = 1.75.
So if the p.i, which is given by prediction, is equal to q.i, the cross entropy is equal to entropy of q, otherwise the cross entropy will be larger than entropy.
