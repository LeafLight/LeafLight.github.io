---
title: CNN Practice Workflow
tags: ["CNN", "Workflow", "machine learning", "MNIST"]
---

## Abstract
Convolution Neural Network is a widely used model all around the world. To have a more profound understanding of it and promote my programming skills with python in a class-style way, I decide to train a CNN used to recognize hand-writing numbers by _MNIST_ dataset.

## Workflow

1. Loading the MNIST data into 2 sets(train, test)
2. Building a CNN, which has a structure of ~4 layers (2 Convolution layers, 2 fully connected layers)~  8 layers(4 Convolution layers, 4 linear full connected layers)
3. Training the network by train set
4. Choosing the parameters by validation set
5. Testing by test set
6. Polishing the network by some tricks(e.g., dropout, pooling, regularization)
## Summary
In this practice, I learned many things that wouldn't be learned just from listenning to online courses.
1. Figure out how `optim`, `nn.Sequential` and some other classes or functions work.
2. Realize how vital the parameters are for the network(_I trained a network with the test acc. of approximately 10% with the help of a learning rate of 0.001, which made me lol_)
3. It surprised me that the network's performance was promoted much more greatly when I just change the `criterion` of loss from `nn.MSELoss` to `nn.CrossEntropyLoss`.

## Something Important
Before actually doing this work code by code, I thought what would impress me most will be the advantage of the CNN. When the result came out, the performance of CNN4 I trained was almost the same as the network with 4 activated linear layers. (~93%,a little higher than the old one's ~90%)

But something that I didn't care about made this practice more interesting.Because of some mistakes, I used MSE as loss of the network instead of cross-entropy, which is recommended for classification problems. After solving the mistakes of using cross-entropy and applying it to my new network, it surprised me with its brilliant performance promotion, of which the acc. reached __93%__ in the __first__ epoch of training. And the final acc. after 8 epochs reached __97%__. It is amazing, because the network of the same structure with MSE has an acc. of about __30%__ after the first epoch, and can __rarely__ reach 95%.

In a blog about cross entropy I wrote before I said that maybe it is just something interesting but not very important since we have MSE, which it more simple and explicit. I would have to change it right nowwwwww.

It also comes to me that neural network and the mathematics behind it are not as simple as I thought before.(Obviously)

## Code
The codes below contains the network, along with the loading of MNIST data and the training of it.
Some negligiable codes are not presented here.(like `one_hot`)
This one used MSE as loss.
```python
#!/bin/python3
#-*-coding:UTF-8-*-
#Author:LeafLight
#Date: 2022-02-24
########################################
import torch
from torch import nn
from torch.nn import functional as F
from MNIST_utils import  one_hot
import torchvision
# step1. load dataset
batch_size= 512
dataset_MU = 0.1307
dataset_SIG = 0.3081


train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist.data', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (dataset_MU,),(dataset_SIG,))
                                        ])
        ),
        batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist.data/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (dataset_MU,),(dataset_SIG,))
                                        
                                        ])
        ),
    batch_size=batch_size, shuffle=False)

sample_x, sample_y = next(iter(train_loader))
print(sample_x.shape, sample_y.shape)
#Output[0]:torch.Size([512, 1, 28, 28]) torch.Size([512])

#step2. build network
############################################################
#<<< network <<<

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4, self).__init__()

        self.model = nn.Sequential(
                #nn.BatchNorm1d(1),
                nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(6, 6, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(6, 1, kernel_size=1, stride=1, padding=0),
                #flatten process required,
                #nn.Flatten(start_dim=2, end_dim=3),
                nn.Flatten(),
                nn.Linear(784, 500),
                nn.ReLU(inplace=True),
                nn.Linear(500, 400),
                nn.ReLU(inplace=True),
                nn.Linear(400, 200),
                nn.ReLU(inplace=True),
                nn.Linear(200, 10),
                )
    def forward(self, x):
        out = self.model(x)
        return out

net = CNN4()
optimizer = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.8)
criterion = nn.MSELoss()
#>>> network >>>
############################################################
#criterion = nn.CrossEntropyLoss()
train_loss = []
#step3. train

for epoch in range(8):
    for batch_idx, (x, y) in enumerate(train_loader):
        #x = x.view(x.size(0), 28*28)
        out = net(x)
        y_onehot = one_hot(y)
        loss = criterion(out, y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx%20==0:
            print("epoch: ", epoch,", batch_idx: ", batch_idx, ", loss: ", loss.data)
        train_loss.append(loss.item())

    total_correct = 0
    for idx,(x, y) in enumerate(test_loader):
            out = net(x)
            pred = out.argmax(dim=1)
            correct = pred.eq(y).sum().float()
            total_correct += correct
            if idx%100 ==0:
                print("idx: ", idx, ", correct: ", correct)
                print("pred/num:", pred[0], y[0])
    total_num = len(test_loader.dataset)
    print("total num: ", total_num)
    print("total correct: ", total_correct)
    print("test acc. :", total_correct/total_num)



```

The one using cross-entropy is changed from the script above with modification below.
```sh
73c73
< criterion = nn.MSELoss()
---
> criterion = nn.CrossEntropyLoss()
83,85c83,85
<         out = net(x)
<         y_onehot = one_hot(y)
<         loss = criterion(out, y_onehot)
---
>         logits = net(x)
>         #y_onehot = one_hot(y)
>         loss = criterion(logits, y)
```
