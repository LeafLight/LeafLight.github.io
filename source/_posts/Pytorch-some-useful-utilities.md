---
title: "[Pytorch] Some useful utilities"
tags: ["MachineLearning", "Pytorch", "Utilities"]
date: 2022-08-16 18:23:52
---

## Reference
[Pytorch Doc](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split)

## Background

I tried to improve my project _BiGCAT_ by doing dataset split today,and when looking up the official documents for the instruction of using `torch.utils.data.random_split()`, other utilities provided in the same page with it  caught my eyes.

## Useful Utilities

### Dataset Types
Other than the type I have known before(_map-style dataset_), here is another one tyep of `Dataset`.

1. __Map-style dataset__: Implements of  `__len__()` and `__getitem__()` protocol are required. It represents a map from keys/indices to samples and can be accessed by `dataset[idx]`.
2. [__Iterable-style dataset__](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset): It is an instance of the subclass `IterableDataset`. We use it where random read are expensive or even improbable and where the batch size depends on the fetched data. `iter(dataset)` can return a stream of data reading from a database, a remote server or even logs generated in real time.

## Working with `collate_fn`
It behaves differently when automatic batching is enabled or disabled.
- When automatic batching enabled: It simply converts every individual Numpy array to Pytorch tensors.
- When automatic batching disabled: It converts a list of tuples into a single tuple of tensors(of course, into tensors).

> If you run into a situation where the outputs of DataLoader have dimensions or type that is different from your expectation, you may want to check your `collate_fn`.

__Something interesting:__ `collate_fn` can even help padding sequence of various length. Here is the [instruction](https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/).

## Tensor Dataset
It is convenient when the task is simple.
Here is an example:
```python
 ds = torch.uitls.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(ds, batch_size=18, shuffle=True, drop_last=True)
```

## Concatenate Datasets
Use `torch.utils.data.ConcatDataset(datasets)`, in which `datasets` is the list of datasets to be concatenated.

## Chain Datasets
Use `torch.utils.data.ChainDataset(datasets)`, which is just like `ConcatDataset()` above but for __Iterable-style Datasets__.

## Subset of Dataset
Use `torch.utils.data.Subset(dataset, indices)`.

## Default Collate
Use `torch.utils.data.default_collate()` to do collation in the default way.

## Split Dataset
Use `torch.utils.data.random_split(dataset, length, generator=<torch._C.Generator object>)`.
Here is an example:
```python
torch.utils.data.random_split(range(0, 10), [3, 7], generator=torch.Generator().manual_seed(42))
```

- __dataset__: the Dateset to split
- __length__: the lengths of splits to be produced
- __generator__: Generator used for the random permutation.

## Sampler
The `DataLoader` we use is consisted of a `Dataset` and a `Sampler`.
Maybe I won't go deep in this subclass in a short time because most of the job I do does not require me to customize a `DataLoader` in details.
