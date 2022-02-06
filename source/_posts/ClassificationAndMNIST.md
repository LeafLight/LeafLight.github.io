# Classification and MNIST dataset

## About the input:

It is a famous dataset of Human-writed number pictures used for deep learning.

 * size of one picture: 28x28(pixels)
 * quantity: 7000 per Number(0~9) (7kx10)

## About the output:

If we simply use 0~9 to represent the result of prediction of our model, it results in a problem that 0~9 are quantitive variables. To avoid it, Here is a new method called __One-Hot__, which is useful in classification.

### One-Hot

#### Representation

It is an special encoding method, which use a matrix(usually has an dimension of 1xn) to represent a categorical variable.

Here comes an example:
```python
red = [1,0,0]
blue = [0,1,0]
yellow = [0,0,1]

#So we can use [1,0,0,0,0,0,0,0,0,0] to represent '0'
```

#### Evaluation(_loss_ calculation)

As to calculate the _loss_ of the model, _Euclidean distance_ is used(because the result of One-Hot Matrix in this model has a dimension of 10x1, which means _Euclidean distance_ is proper).

_here remains a question-- why Euclidean distance is not proper for all kinds of data_

Here comes an example:
```python
Real_label = [1, 0, 0, 0]
Pred = [0.9, 0.1, 0.1, 0.1]
loss =sum([pow(Real_label[i]-Pred[i],2) for i in range(0,4)]) 
```

## About the model:

### Sigmoid function and ReLU function:

* Sigmoid function:
![Sigmoid function](https://www.google.com/imgres?imgurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1200%2F1*a04iKNbchayCAJ7-0QlesA.png&imgrefurl=https%3A%2F%2Fmedium.com%2F%40toprak.mhmt%2Factivation-functions-for-deep-learning-13d8b9b20e&tbnid=Zl9O5xKIlo6tvM&vet=12ahUKEwjH99qG-tb1AhXYqnIEHciXB-oQMygBegUIARDaAQ..i&docid=HgiWI3njmHtosM&w=1200&h=630&itg=1&q=Sigmoid%20function&hl=en-US&client=ubuntu&ved=2ahUKEwjH99qG-tb1AhXYqnIEHciXB-oQMygBegUIARDaAQ)

$$ h_ \theta (x) =  \frac{\mathrm{1} }{\mathrm{1} + e^- \theta^Tx }  $$
* ReLU function:
![ReLU function](https://www.google.com/imgres?imgurl=https%3A%2F%2Fmiro.medium.com%2Fmax%2F357%2F1*oePAhrm74RNnNEolprmTaQ.png&imgrefurl=https%3A%2F%2Fmedium.com%2F%40kanchansarkar%2Frelu-not-a-differentiable-function-why-used-in-gradient-based-optimization-7fef3a4cecec&tbnid=0UdUiZ4X2VLDiM&vet=12ahUKEwifmaTS-tb1AhWHq3IEHS10CgoQMygBegUIARC-AQ..i&docid=8NiVbpcoDLL_LM&w=357&h=278&itg=1&q=ReLU%20function&hl=en-US&client=ubuntu&ved=2ahUKEwifmaTS-tb1AhWHq3IEHS10CgoQMygBegUIARC-AQ)

$$ R(x) = max(0,x) $$

_Why they are special: We don't use linear function because linear factor is not vey qualified to deal with complex REAL-WORLD problem(Though recognizing a hand-writing number is easy for us human beings, it is difficult to teach the machine this technique.). Sigmoid F and ReLU F have a better simulation of the nerves of human beings, which have thresholds._ 

### Gradient Descent
* $$ Objective = \sum (pred - Y) $$

* minimize objective 

* for a new $$X$$

	* $$[W_1,W_2,W_3]$$

	* $$[b_1,b_2,b_3]$$

	* $$pred = W_3 * \left{ W_2 \left[ W_1X + b_1\right] +b_2\right} + b_3$$

	* $$argmax(pred)$$

### Code

MNIST_utils.py 
```python
import torch
from matplotlib import pyplot as plt

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(img, label, name, num=(4,4), dataset_MU=0.1307, dataset_SIG=0.3081):
    fig = plt.figure()
    for i in range(0, num[0]*num[1]):
        plt.subplot(num[0], num[1], i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*dataset_SIG+dataset_MU, cmap='gray', interpolation='none')
        plt.title("{}:{}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1, index=idx, value=1)
    return out
```

MNIST_train.py
```python
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt

from MNIST_utils import plot_image, plot_curve, one_hot

batch_size = 512
dataset_MU = 0.1307
dataset_SIG = 0.3081

# step1. load dataset
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
plot_image(sample_x,sample_y,'image sample',num=(5,5))
#plot_image() is the function imported from MNIST_utils.py
#Output[1]:some hand-writing numbers' pictures(default 10)

# step2. Network model building
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # wx+b
        self.fc1 = nn.Linear(28*28, 256)
        # 28*28: the pixel size of one picture in the dataset
        # 256: hyper para.
        self.fc2 = nn.Linear(256, 64)
        # 256: the size of output of the prior layer
        # 64: hyper para.
        self.fc3 = nn.Linear(64,10)
        # 64: the size of output of the prior layer
        # 10: the size of final output
    def forward(self, x):
        #x: [b, 1, 28, 28]
        #h1 = relu(w1x+b1)
        x = F.relu(self.fc1(x))
        #h2 = relu(w2h1+b2)
        x = F.relu(self.fc2(x))
        #h3 = (w3h2+b3)
        x = self.fc3(x)

        return x
#initialize the network
net = Net()
#[w1, w2, w3, b1, b2, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9 )
#para. used to plot the loss_step plot
train_loss = []
# step3. Training
for epoch in range(3):
    #3 times of traversal of the train set
    for batch_idx, (x, y) in enumerate(train_loader):
        #print(x.shape, y.shape)
        #Output[2]: x:[b, 1, 28, 28], y:[512]
        # [b, 1, 28, 28] => [b, feature]
        x = x.view(x.size(0), 28*28)
        # =>[b, 10]
        out = net(x)
        # [b, 10]
        y_onehot = one_hot(y)
        # loss = mse(out, y_onehot)
        loss = F.mse_loss(out, y_onehot)
        # clean the gradient
        optimizer.zero_grad()
        # calculate the gradient
        loss.backward()
        # update the gradient
        #w' = w- lr*grad
        optimizer.step()
        #para. 'loss' is a tensor object, use .item() to convert it to numpy object
        train_loss.append(loss.item())
        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())
# we get the optimal [w1, b1, w2, b2, w3, b3]
plot_curve(train_loss)

# step4. test
total_correct = 0
for x,y in test_loader:
    x = x.view(x.size(0), 28*28)
    out = net(x)
    #out: [b, 10] => pred: [b]
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float()
    total_correct += correct

total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)

x, y = next(iter(test_loader))
out = net(x.view(x.size(0),28*28))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')

```
