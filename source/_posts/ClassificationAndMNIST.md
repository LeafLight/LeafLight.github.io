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
