---
title: Custom Dataset--Pokemon dataset loading by Pytorch
tags: ["Pytorch", "Pokemon", "Dataset", "WorkFlow"]
category:
- [MachineLearning]
---

## Reference

Method and Data from:
[Bilibili Online Course of Pytorch by Liangqu Long-p99](https://www.bilibili.com/video/BV1fT4y1d7av?p=99)

## Abstract

In the practice of kinds of neural networks before, data used to train the networks is provided by "MNIST" or "CIFAR" and so on, which can be loaded by pytorch easily. The convenice of loading results from the powerful pytorch utilities.

But more often than not, we need to use our own dataset, which means there isn't any completed utilities provided by pytorch utilities.

Some might say that we can load the data by our own scripts, which just needs to do the work of dataloading, shuffling, train-val-test slicing, batch seperation and so on.(In my recent work about SMILEs, I used this 'mannual' method.)

To the data that restored in one single file like a csv file, it is easy. But when the data structure is more complex like images which are restored in differnt files in different filefold, the 'manual' way may be a little troublesome.

Fortunately, Pytorch provides a useful `class` called `Dataset` used to load the data.

## torch.utils.data.Dataset

Just like `torchvision.datasets.MNIST` or some other datasets supported by Pytorch, the dataset we used can satisfy most of the functions we need only if we create a class inheriting from `torch.utils.data.Dataset` for it.

Let's use the dataset of Pokemon as example, which has a structure of:
```
Pokemon(Filefold)-------|- Pokemon1 ----|-picture1.jpg
			|		|-picture2.png
			|		|-...
			|
			|- Pokemon2
			|- Pokemon3
			|- ...
```

Import:
```python
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
import random
import csv
```

### Inherit from `Dataset`

```python
class Pokemon(Dataset):
	def __init__(self, root, resize, mode):
		super(Pokemon, self).__init__()
		pass
	def __len__(self,):
		pass
	def __getitem__(self):
		pass

```

In the codes above, three methods of the class `Pokemon` were defined. They are the methods that are necessary to realize. The explanation of them:
1. `__init__`: Some initialization work
2. `__len__`: return the length of the dataset
3. `__getitem__`: return the item of specific index from the dataset

### Initialization

To realize `__len__` and `__getitem__` easily, proper initialization is important. And we need to realize some simple options when initialization. You may notice there are some parameters in the codes above like `root`. Here are their explanation:
1. `root`: the root dir of the dataset
2. `resize`: the size to transform the image in the dataset into
3. `mode`: 'train', 'val' or 'test'

1. self.xxx
```python
	...
	def __init___(self, root, resize, mode):
		super(Pokemon, self).__init__()
		self.root = root
		self.resize = resize
		self.mode = mode
	
		self.name2label = {}
	...
```

2. self.name2label
```python
	...
	# a dict used to store the mapping of name and label
	self.name2label = {}
	# loop to fill the mapping
	for name in sorted(os.listdir(os.path.join(root))):
		# skip the non-filefold file
		if not os.path.isdir(os.path.join(root, name)):
			continue
		self.name2label[name] = len(self.name2label.keys())
		# label one by one
	...
```

3. load the (image, label) pairs
```python
	...
		#lable one by one
	self.images, self.labels = self.load_csv('images.csv')
	...
```
`self.load_csv` is a auxiliary method
```python
	def load_csv(self, filename):
		#load all the images directly may abuse the cpu
		#loop over all the types and load their image_path one by one
		#create the csv if not existing
		if not os.path.exists(os.path.join(self.root, filename)):
			images = []
			for name in self.name2label.keys():
				images += glob.glob(os.path(self.root, name, "*.png"))
				images += glob.glob(os.path(self.root, name, "*.jpg"))
				images += glob.glob(os.path(self.root, name, "*.jpeg"))
			# len: 1167, '/pokemon\\bulbasaur\\0000000.png'
			# write to the csv file
			# shuffle the images
			random.shuffle(images)
			# weite to the csv file
			with open(os.path.join(self.root, filename), mode='w', newline='') as f:
				writer = csv.writer(f)
				label = self.name2label[name]
				# 'pokemon\\bulbasaur\\00000.png', 0
				writer.writerow([img, label])
				print("write into csv file:", filename)
		images, labels = [], []
		# read the csv if it exists
		with open(os.path.join(self.root, filename)) as f:
			print("read the csv file:", filename)
			reader = csv.reader()
			for row in reader:
				img, label = row
				label = int(label)
				image.append(img)
				labels.append(label)
		assert len(images) == len(labels)
		return images, labels
```
In the codes above, the feature of shuffling is realized by `random.shuffle(images)` easily. It is because of the fact that the labels are contained in the path of the images. In other cases, in which images and labels are seperated, `zip()` or randomize the index may solve the problem.

3. length of the data
```python
	...
	def __len__(self):
		return len(self.images)
	...
```

4. get the item
```python
	...
	def __getitem__(self, idx):
	# idx: 0~len(images)
	# self.images, self.labels
	img, label = self.images[idx], self.labels[idx]

	tf = transforms.Compose([
		lambda x : Image.open(x).convert('RGB'), #img path => data
		transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
		transforms.RandomRotation(15),
		transforms.CenterCrop(self.resize),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225]),
	])
	
	img = tf(img)
	label = torch.tensor(label)
	return img, label
```

5. to visualize
```python
	...
	def denormalize(self, x_hat):
		# denormalize for  visualization
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		
		# x_hat= (x - mean)/std
		# x = x_hat * std + mean
		# x: [c, h, w]
		# mean: [3] => [3, 1, 1]
		mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
		std = torch.tensor(std).unsqueeze(1).unsqueeze(1)


		x = x_hat * std + mean

		return x
	...
```
