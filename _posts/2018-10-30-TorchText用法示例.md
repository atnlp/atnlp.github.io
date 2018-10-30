---
title: TorchText用法示例
description: TorchText的一般用法，灵活的Dataset定义和迭代器使用方法总结。
categories:
 - PyTorch
tags:
---
<font size='4'>

[toc] 
# TorchText
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最近开始使用PyTorch进行NLP神经网络模型的搭建，发现了torchtext这一文本处理神奇，但是因为nlp的热度远不如cv，对于torchtext介绍的相关博客数量也远不如torchvision。在使用过程中主要参考了[A Comprehensive Introduction to Torchtext](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)和[Language modeling tutorial in torchtext](http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/)这两篇博客和[torchtext官方文档](https://torchtext.readthedocs.io/en/latest/index.html)，对于torchtext的基本用法有了大致的了解。在以上两篇博客的基础上，本文主要介绍torchtext中一些更灵活的数据处理方式。

## torchtext概述
从第一篇参考博客中可以发现，torchtext对数据的处理可以概括为Field，Dataset和迭代器这三部分。
### Field对象
> Field对象指定要如何处理某个字段.
### Dataset
> Dataset定义数据源信息.
### 迭代器
> 迭代器返回模型所需要的处理后的数据.迭代器主要分为Iterator, BucketIerator, BPTTIterator三种。
- Iterator：标准迭代器
- BucketIerator：相比于标准迭代器，会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。
- BPTTIterator

## 自定义Dataset类
> 当对原始数据集只进行简单的划分处理时，例如读取数据并划分训练集验证集等操作，可以直接使用TabularDataset类和split类方法来实现，该类支持读取csv,tsv等格式。但是当我们需要对数据进行更多的预处理时，例如shuffle，dropout等数据增强操作时，自定义Dataset会更灵活。

实验数据集仍然使用[A Comprehensive Introduction to Torchtext](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)中使用的小批量数据集，为了简化代码，只保留了toxic这一个标签列。
![](https://ws4.sinaimg.cn/large/006tNbRwly1fwpg9xk8skj30t007gmyb.jpg)

- 核心代码如下
```python
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import random
import os


train_path = 'data/train_one_label.csv'
valid_path = "data/valid_one_label.csv"
test_path = "data/test.csv"

# 定义Dataset
class MyDataset(data.Dataset):

    def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
        fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", text_field), ("toxic", label_field)]
        
        examples = []
        csv_data = pd.read_csv(path)
        print('read data from {}'.format(path))

        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(csv_data['comment_text']):
                examples.append(data.Example.fromlist([None, text, None], fields))
        else:
            for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
                if aug:
                    # do augmentation
                    rate = random.random()
                    if rate > 0.5:
                        text = self.dropout(text)
                    else:
                        text = self.shuffle(text)
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([None, text, label - 1], fields))
        # 之前是一些预处理操作，此处调用super调用父类构造方法，产生标准Dataset
        super(GrandDataset, self).__init__(examples, fields, **kwargs)

    def shuffle(self, text):
        text = np.random.permutation(text.strip().split())
        return ' '.join(text)

    def dropout(self, text, p=0.5):
        # random delete some text
        text = text.strip().split()
        len_ = len(text)
        indexs = np.random.choice(len_, int(len_ * p))
        for i in indexs:
            text[i] = ''
        return ' '.join(text)

```
 我们定义了一个MyDataset类，该类继承了torchtext.data.Dataset类，并在init方法内实现了数据和fields参数的绑定。在类内部也实现了shuffle和dropout两个数据预处理方法。最后使用super调用父类的标准init方法，实现标准torchtext中的标准Dataset.
 
 data.Example返回单个样本，提供了fromCSV和fromJSON等多个方法，可以从多种形式构建Dataset所需的标准数据。
 此外，对于像id这种在模型训练中不需要的特征，在构建Dataset的过程中可以直接使用None来代替。

- 构建MyDataset对象
![](https://ws3.sinaimg.cn/large/006tNbRwly1fwqc2b2qyej31kw01q3yz.jpg)
可以查看train的相关信息
![](https://ws3.sinaimg.cn/large/006tNbRwly1fwqc1jq85cj31kw06bdhh.jpg)

## 构建迭代器
```python
from torchtext.data import Iterator, BucketIterator
train_iter, val_iter = BucketIterator.splits(
        (train, valid), # we pass in the datasets we want the iterator to draw data from
        batch_sizes=(8, 8),
        device=-1, # if you want to use the GPU, specify the GPU number here
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)
test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
```
我们已经了解到BucketIterator相比Iterator的优势是会自动选取样本长度相似的数据来构建批数据。但是在测试集中一般不想改变样本顺序，因此测试集使用Iterator迭代器来构建。<br>sort_within_batch参数设置为True时，按照sort_key按降序对每个小批次内的数据进行排序。如果我们需要padded序列使用pack_padded_sequence转换为PackedSequence对象时，这是非常重要的，我们知道如果想pack_padded_sequence方法必须将批样本按照降序排列。由此可见，torchtext不仅可以对文本数据进行很方变的处理，还可以很方便的和torchtext的很多内建方法进行结合使用。<br>实验选取的训练集共有25条样本
![](https://ws4.sinaimg.cn/large/006tNbRwgy1fwqcbe2gnfj31kw03bq35.jpg)
我们接下来可以利用python内建的iter来查看构建的迭代器信息。
![](https://ws1.sinaimg.cn/large/006tNbRwgy1fwqc8hn5kqj31kw06zdh9.jpg)
由输出结果可知，对于每一批数据，迭代器会将样本长度构建为统一的长度，对于第一批数据的comment_text特征，迭代器统一构建成了[71*8]的维度。
### 批数据的使用
![](https://ws3.sinaimg.cn/large/006tNbRwgy1fwqceujf6gj31kw0ov0zt.jpg)
使用迭代器构建批数据后，我们可以直接使用python的for循环来遍历，观察输出结果，对于共有25条数据的训练集，torchtext的迭代器将训练集构建成了四批数据。在此基础上我们可以进而将数据喂给模型。

## 完整代码
- 完整代码见我的github仓库
</font>