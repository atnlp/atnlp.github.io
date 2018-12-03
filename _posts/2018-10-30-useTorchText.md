---
title: TorchText用法示例
description: TorchText的一般用法，灵活的Dataset定义和迭代器使用方法总结。文末附完整代码。
categories:
 - PyTorch
tags:
---

# TorchText
> &nbsp;&nbsp;&nbsp;&nbsp;最近开始使用PyTorch进行NLP神经网络模型的搭建，发现了torchtext这一文本处理神器，可以方便的对文本进行预处理，例如截断补长、构建词表等。但是因为nlp的热度远不如cv，对于torchtext介绍的相关博客数量也远不如torchvision。在使用过程中主要参考了[A Comprehensive Introduction to Torchtext](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)和[Language modeling tutorial in torchtext](http://mlexplained.com/2018/02/15/language-modeling-tutorial-in-torchtext-practical-torchtext-part-2/)这两篇博客和[torchtext官方文档](https://torchtext.readthedocs.io/en/latest/index.html)，对于torchtext的基本用法有了大致的了解。在以上两篇博客的基础上，本文对torchtext的使用做一个概括性的总结。文末附完整代码。

## torchtext概述
&nbsp;&nbsp;&nbsp;&nbsp;torchtext对数据的处理可以概括为Field，Dataset和迭代器这三部分。
### Field对象
> Field对象指定要如何处理某个字段.

### Dataset
> Dataset定义数据源信息.

### 迭代器
> 迭代器返回模型所需要的处理后的数据.迭代器主要分为Iterator, BucketIerator, BPTTIterator三种。

- Iterator：标准迭代器
- BucketIerator：相比于标准迭代器，会将类似长度的样本当做一批来处理，因为在文本处理中经常会需要将每一批样本长度补齐为当前批中最长序列的长度，因此当样本长度差别较大时，使用BucketIerator可以带来填充效率的提高。除此之外，我们还可以在Field中通过fix_length参数来对样本进行截断补齐操作。
- BPTTIterator: 基于BPTT(基于时间的反向传播算法)的迭代器，一般用于语言模型中。

## 使用Dataset类
&nbsp;&nbsp;&nbsp;&nbsp;实验数据集仍然使用[A Comprehensive Introduction to Torchtext](http://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/)中使用的小批量数据集，为了简化代码，只保留了toxic这一个标签列。

- 查看数据集
![](https://ws4.sinaimg.cn/large/006tNbRwly1fwpg9xk8skj30t007gmyb.jpg)
- 导入torchtext相关包

```python
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
```
- 构建Field对象

```python
tokenize = lambda x: x.split()
# fix_length指定了每条文本的长度，截断补长
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = data.Field(sequential=False, use_vocab=False)
```

- 使用torchtext内置的Dataset构建数据集

&nbsp;&nbsp;&nbsp;&nbsp;torchtext预置的Dataset类的API如下，我们必须至少传入examples和fields这两个参数。examples为由torchtext中的Example对象构造的列表，Example为对数据集中一条数据的抽象。fields可简单理解为每一列数据和Field对象的绑定关系，在下面的代码中将分别用train\_examples和test\_examples来构建训练集和测试集的examples对象，train\_fields和test\_fields数据集的fields对象。

> class torchtext.data.Dataset(examples, fields, filter_pred=None)

```python
# 读取数据
train_data = pd.read_csv('data/train_one_label.csv')
valid_data = pd.read_csv('data/valid_one_label.csv')
test_data = pd.read_csv("data/test.csv")
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False)

# get_dataset构造并返回Dataset所需的examples和fields
def get_dataset(csv_data, text_field, label_field, test=False):
	# id数据对训练在训练过程中没用，使用None指定其对应的field
    fields = [("id", None), # we won't be needing the id, so we pass in None as the field
                 ("comment_text", text_field), ("toxic", label_field)]       
    examples = []

    if test:
        # 如果为测试集，则不加载label
        for text in tqdm(csv_data['comment_text']):
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in tqdm(zip(csv_data['comment_text'], csv_data['toxic'])):
            examples.append(data.Example.fromlist([None, text, label], fields))
    return examples, fields

# 得到构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset(train_data, TEXT, LABEL)
valid_examples, valid_fields = get_dataset(valid_data, TEXT, LABEL)
test_examples, test_fields = get_dataset(test_data, TEXT, None, test=True)

# 构建Dataset数据集
train = data.Dataset(train_examples, train_fields)
valid = data.Dataset(valid_examples, valid_fields)
test = data.Dataset(test_examples, test_fields)
```
&nbsp;&nbsp;&nbsp;&nbsp;data.Example返回单个样本，提供了fromCSV和fromJSON等多个方法，可以从多种形式构建Dataset所需的标准数据。<br>
 &nbsp;&nbsp;&nbsp;&nbsp;此外，对于像id这种在模型训练中不需要的特征，在构建Dataset的过程中可以直接使用None来代替。特别注意的是，对于test中的label，在机器学习比赛中我们不知道最终的测试集的标签，因此此处在构建fields和examples时都相应的设置成了None，如果是在自己划分出来的测试集，此时测试集也有对应的标签label，需要修改对应代码，用对应的Field项替换None.

## 自定义Dataset类
> 当构建简单的数据集时，可直接使用torch.text.Dataset来构建，当对原始数据集只进行简单的划分处理时，例如读取数据并划分训练集验证集等操作，也可以直接使用TabularDataset类和split类方法来实现，该类支持读取csv,tsv等格式。但是当我们需要对数据进行更多的预处理时，例如shuffle，dropout等数据增强操作时，自定义Dataset会更灵活。

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
#定义Dataset
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
        # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super(MyDataset, self).__init__(examples, fields, **kwargs)

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

&nbsp;&nbsp;&nbsp;&nbsp;我们定义了一个MyDataset类，该类继承了torchtext.data.Dataset类，并在init方法内实现了数据和fields参数的绑定。在类内部也实现了shuffle和dropout两个数据预处理方法。最后使用super初始化父类，实现torchtext中的标准Dataset.
 

&nbsp;&nbsp;&nbsp;&nbsp;构建MyDataset对象
![](https://ws3.sinaimg.cn/large/006tNbRwly1fwqc2b2qyej31kw01q3yz.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;因为test数据集没有标签label，必须制定label_field为None
![](https://ws1.sinaimg.cn/large/006tNbRwgy1fxop6j4owcj31ou01s74p.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;可以查看train的相关信息
![](https://ws3.sinaimg.cn/large/006tNbRwly1fwqc1jq85cj31kw06bdhh.jpg)

## 构建迭代器
> 在训练神经网络时，是对一个batch的数据进行操作，因此我们还需要使用torchtext内部的迭代器对数据进行处理。 

```python
from torchtext.data import Iterator, BucketIterator
# 若只针对训练集构造迭代器
# train_iter = data.BucketIterator(dataset=train, batch_size=8, shuffle=True, sort_within_batch=False, repeat=False)

# 同时对训练集和验证集进行迭代器的构建
train_iter, val_iter = BucketIterator.splits(
        (train, valid), # 构建数据集所需的数据集
        batch_sizes=(8, 8),
        device=-1, # 如果使用gpu，此处将-1更换为GPU的编号
        sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
        sort_within_batch=False,
        repeat=False # we pass repeat=False because we want to wrap this Iterator layer.
)

test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)
```
&nbsp;&nbsp;&nbsp;&nbsp;BucketIterator相比Iterator的优势是会自动选取样本长度相似的数据来构建批数据。但是在测试集中一般不想改变样本顺序，因此测试集使用Iterator迭代器来构建。<br>&nbsp;&nbsp;&nbsp;&nbsp;sort\_within\_batch参数设置为True时，按照sort\_key按降序对每个小批次内的数据进行排序。如果我们需要padded序列使用pack\_padded\_sequence转换为PackedSequence对象时，这是非常重要的，我们知道如果想pack\_padded\_sequence方法必须将批样本按照降序排列。由此可见，torchtext不仅可以对文本数据进行很方变的处理，还可以很方便的和torchtext的很多内建方法进行结合使用。<br>

&nbsp;&nbsp;&nbsp;&nbsp;实验选取的训练集共有25条样本
![](https://ws4.sinaimg.cn/large/006tNbRwgy1fwqcbe2gnfj31kw03bq35.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;我们接下来可以利用python内建的iter来查看构建的迭代器信息。
![](https://ws3.sinaimg.cn/large/006tNbRwly1fxq4iohofyj31da0963zu.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;由输出结果可知，对于每一批数据，迭代器会将样本长度构建为统一的长度，对于第一批数据的comment_text特征，迭代器统一构建成了[200*8]的维度。因为在构建TEXT的时候通过fix_length=200指定了序列的长度。

### 批数据的使用

![](https://ws1.sinaimg.cn/large/006tNbRwgy1fxq4kcoc38j31d60podm2.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;使用迭代器构建批数据后，我们可以直接使用python的for循环来遍历，观察输出结果，对于共有25条数据的训练集，torchtext的迭代器将训练集构建成了四批数据。在此基础上我们可以进而将数据传入模型。

## 构建词表
> 所谓构建词表，即需要给每个单词编码，也就是用数字来表示每个单词，这样才能够传入模型中。

### 最简单的方式，bulid_vocab()方法中传入用于构建词表的数据集

	TEXT.build_vocab(train)

### 使用预训练的词向量
> 在使用pytorch或tensorflow等神经网络框架进行nlp任务的处理时，可以通过对应的Embedding层做词向量的处理，更多的时候，使用预训练好的词向量会带来更优的性能，下面介绍如何在torchtext中使用预训练的词向量，进而传送给神经网络模型进行训练。

#### 方式1：使用torchtext默认支持的预训练词向量

&nbsp;&nbsp;&nbsp;&nbsp;默认情况下，会自动下载对应的预训练词向量文件到当前文件夹下的.vector\_cache目录下，.vector\_cache为默认的词向量文件和缓存文件的目录。

```python
from torchtext.vocab import GloVe
from torchtext import data
TEXT = data.Field(sequential=True)
# 以下两种指定预训练词向量的方式等效
# TEXT.build_vocab(train, vectors="glove.6B.200d")
TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
# 在这种情况下，会默认下载glove.6B.zip文件，进而解压出glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt这四个文件，因此我们可以事先将glove.6B.zip或glove.6B.200d.txt放在.vector_cache文件夹下(若不存在，则手动创建)。
```
#### 方式2：使用外部预训练好的词向量

&nbsp;&nbsp;&nbsp;&nbsp;上述使用预训练词向量文件的方式存在一大问题，即我们每做一个nlp任务时，建立词表时都需要在对应的.vector_cache文件夹中下载预训练词向量文件，如何解决这一问题？我们可以使用torchtext.vocab.Vectors中的name和cachae参数指定预训练的词向量文件和缓存文件的所在目录。因此我们也可以使用自己用word2vec等工具训练出的词向量文件，只需将词向量文件放在name指定的目录中即可。

- 通过name参数可以指定预训练的词向量文件所在的目录

&nbsp;&nbsp;&nbsp;&nbsp;默认情况下预训练词向量文件和缓存文件的目录位置都为当前目录下的 .vector\_cache目录，虽然通过name参数指定了预训练词向量文件存在的目录，但是因为缓存文件的目录没有特殊指定，此时在当前目录下仍然需要存在 .vector\_cache 目录。

```python
# glove.6B.200d.txt为预先下载好的预训练词向量文件
if not os.path.exists(.vector_cache):
    os.mkdir(.vector_cache)
vectors = Vectors(name='myvector/glove/glove.6B.200d.txt')
TEXT.build_vocab(train, vectors=vectors)
```

- 通过cache参数指定缓存目录

```python
# 更进一步的，可以在指定name的同时同时指定缓存文件所在目录，而不是使用默认的.vector_cache目录
cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
vectors = Vectors(name='myvector/glove/glove.6B.200d.txt', cache=cache)
TEXT.build_vocab(train, vectors=vectors)
```

### 在模型中指定Embedding层的权重

&nbsp;&nbsp;&nbsp;&nbsp;在使用预训练好的词向量时，我们需要在神经网络模型的Embedding层中明确地传递嵌入矩阵的初始权重。权重包含在词汇表的vectors属性中。以Pytorch搭建的Embedding层为例：

```
# 通过pytorch创建的Embedding层
embedding = nn.Embedding(2000, 256)
# 指定嵌入矩阵的初始权重
weight_matrix = TEXT.vocab.vectors
embedding.weight.data.copy_(weight_matrix )
# 指定预训练权重的同时设定requires_grad=True
# embeddings.weight = nn.Parameter(embeddings, requires_grad=True)
```

## 使用torchtext构建的数据集用于LSTM
> 因数据集太小，无法收敛，只作为demo熟悉torchtext和pytorch之间的用法

- 核心代码如下

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
weight_matrix = TEXT.vocab.vectors

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 300)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=300, hidden_size=128, num_layers=1)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128, 2)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out = self.lstm(embeds)[0]  # lstm_out:200x8x128
        # 取最后一个时间步
        final = lstm_out[-1]  # 8*128
        y = self.decoder(final)  # 8*2 
        return y

def main():
	model = LSTM()
	model.train()
	optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
	loss_funtion = F.cross_entropy
	
	for epoch, batch in enumerate(train_iter):
	    optimizer.zero_grad()
	    start = time.time()
	    # text = batch.text.permute(1, 0)
	    predicted = model(batch.comment_text)
	
	    loss = loss_funtion(predicted, batch.toxic)
	    loss.backward()
	    optimizer.step()
	    print(loss)
	    
if __name__ == '__main__':
    main()	    
```

## 说明
&nbsp;&nbsp;&nbsp;&nbsp; 除了使用torchtext之外，也可以使用keras中preprocessing包中的相关方法做数据预处理，再使用torch.utils.data.TensorDataset来构造数据集，使用torch.utils.data.DataLoader来构建迭代器。该部分代码改天来更新。

## 代码示例

### 本文所涉及内容的完整代码

- 完整demo代码见我的github仓库: https://github.com/atnlp/torchtext-summary

### 一个使用torchtext内置数据集的例子
```python
import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
import numpy as np

def load_data(opt):
    # use torchtext to load data, no need to download dataset
    print("loading {} dataset".format(opt.dataset))
    # set up fields
    text = data.Field(lower=True, include_lengths=True, batch_first=True, fix_length=opt.max_seq_len)
    label = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(text, label)

    # build the vocabulary
    text.build_vocab(train, vectors=GloVe(name='6B', dim=300))
    label.build_vocab(train)

    # print vocab information
    print('len(TEXT.vocab)', len(text.vocab))
    print('TEXT.vocab.vectors.size()', text.vocab.vectors.size())
```



