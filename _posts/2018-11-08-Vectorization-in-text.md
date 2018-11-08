---
title: NLP中的文本预处理——向量化
description: 在自然语言处理的一些任务中，我们总是要先将源数据处理成符合模型算法输入的形式，在经过分词、停用词移除等其他操作之后，我们一般需要将文本转化成矩阵的形式，这一步也被称为特征提取（feature extraction）或者向量化（vectorization）。本文简要介绍一下词袋模式(BoW)，tf-idf，Hash，lsa, lda, Doc2vec这几种方法的概念和使用方式。

categories:
 - NLP
tags: NLP Python
---

# 向量化
&nbsp;&nbsp;&nbsp;&nbsp;在自然语言处理的一些任务中，我们总是要先将源数据处理成符合模型算法输入的形式，在经过分词、停用词移除等其他操作之后，我们一般需要将文本转化成矩阵的形式，这一步也被称为特征提取（feature extraction）或者向量化（vectorization）。本文主要介绍词袋模式(BoW)，tf-idf，Hash，lsa, lda, Doc2vec这几种方法。

## 词袋模型

### 概念

&nbsp;&nbsp;&nbsp;&nbsp;词袋模型假定对于一个文档，忽略它的单词顺序和语法、句法等要素，将其仅仅看作是若干个词汇的集合，文档中每个单词的出现都是独立的，不依赖于其它单词是否出现。

&nbsp;&nbsp;&nbsp;&nbsp;对于这样一段语料：

    corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']<br>
可以构建一个长度为9的词表，如下所示

    {'this': 8,
     'is': 3,
     'the': 6,
     'first': 2,
     'document': 1,
     'second': 5,
     'and': 0,
     'third': 7,
     'one': 4}
 
&nbsp;&nbsp;&nbsp;&nbsp;那么对于任意一句话，都可以用长度为9(词表长度)的矩阵来表示，比如This document is the second document.这句话，就可以用[0, 2, 0, 1, 0, 1, 1, 0, 1]向量来表示，矩阵索引对应词表中对应的单词，数字的大小代表该单词在语句中出现的次数, 对于转化时存在的一个新文档，如果这个文档里面的某些词不在已知的词汇表中，那么这些词就会被忽略掉，

### 词袋模型的缺点

&nbsp;&nbsp;&nbsp;&nbsp;词袋模型的一大缺点就是没有考虑词序之间的关系，也没有考虑词语之间的关系。例如：对于good和well这两个单词，在很多情况下有着非常相关的含义，但是在词袋模型的表示下，在计算余弦相似度时的值可能为0.

### BoW完整示例

- 使用sklearn中的CountVectorizer来完成词袋模型

```python
from sklearn.feature_extraction.text import CountVectorizer
# 定义语料库
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
# 创建词袋模型
vectorizer = CountVectorizer()
# 使用给定的语料库来训练该模型
vectorizer.fit_transform(corpus)
# 词表中的每一维相当于一个特征
print(vectorizer.get_feature_names())  # ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
# 查看词表中单词对应的索引
print(vectorizer.vocabulary_) # {'this': 8, 'is': 3, 'the': 6,'first': 2,'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}
# 定义一个测试用例
test = ['This is a apple']
result = vectorizer.transform(test)
# a 和 apple不存在词表中，会直接忽略掉
print(result.toarray())  # array([[0, 0, 0, 1, 0, 0, 0, 0, 1]])
```
&nbsp;&nbsp;&nbsp;&nbsp;在向量化后将数据放进机器学习模型中进行训练时，可以直接放入向量化后的结果，而不必调用toarray()方法转化为向量的形式。

## Tf-idf

### 概念

&nbsp;&nbsp;&nbsp;&nbsp;对于一句话或一片文档张中出现的单词，我们很容易的想到赋予频繁出现的单词一个重要的权重，但是这样会造成一个问题，即像英文中的'this'，'is'或中文中的'的'这样的词在每个文档中都会频繁出现，这种以频率计算权重的方法会赋予这些词很大的权重，而这种频繁出现的单词难以表示一句话或一篇文章所包含的信息。Tf-idf计算频率的基础上加上了逆文本频率指数的概念，有效弥补了单一频率计算权重的缺点。

&nbsp;&nbsp;&nbsp;&nbsp;TF-IDF原本是一种统计方法，用以评估字词对于一个文件集或一个语料库中的其中一份文件的重要程度。这个方法认为，字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降，其实也就相当于在CountVectorizer的基础上结合整个语料库考虑单词的权重，并不是说这个单词出现次数越多它就越重要。核心思想可用下图来表示：
![](http://cloudpicture.oss-cn-hangzhou.aliyuncs.com/18-11-6/66557193.jpg)

1. 词w在文档d中的词频tf (Term Frequency)，即词w在文档d中出现次数count(w, d)和文档d中总词数size(d)的比值：

2. tf(w,d) = count(w, d) / size(d)
词w在整个文档集合中的逆向文档频率idf (Inverse Document Frequency)，即文档总数n与词w所出现文件数docs(w, D)比值的对数:
idf = log(n / docs(w, D))


### Tf-idf完整示例

- 使用sklearn中的TfidfVectorizer来完成Tf-idf向量化

```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
# 创建Tf-idf模型
vectorizer = TfidfVectorizer()
# 使用给定的语料库来训练该模型
X = vectorizer.fit_transform(corpus)
print('Tfidf result of corpus')
print(X.toarray())

# 测试
test = ['This is a apple']
result = vectorizer.transform(test)
print('Tfidf result of test')
print(result.toarray())
```

输出结果如下所示

> skleran中的tfidf模块做了一些其他处理，计算结果不等同于上述公式，但基本思想是一致的！

```python
Tfidf result of corpus
array([[0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524],
       [0.        , 0.6876236 , 0.        , 0.28108867, 0.        ,
        0.53864762, 0.28108867, 0.        , 0.28108867],
       [0.51184851, 0.        , 0.        , 0.26710379, 0.51184851,
        0.        , 0.26710379, 0.51184851, 0.26710379],
       [0.        , 0.46979139, 0.58028582, 0.38408524, 0.        ,
        0.        , 0.38408524, 0.        , 0.38408524]])
        
Tfidf result of test       
array([[0.        , 0.        , 0.        , 0.70710678, 0.        ,
        0.        , 0.        , 0.        , 0.70710678]])
```



## Hash向量化方法

### 概念
&nbsp;&nbsp;&nbsp;&nbsp;词袋模式和Tfidf方法都可以有效的表示矩阵，但是一大缺点是向量化时生成的矩阵严重依赖于词表的长度，上述例子中词表长度只为9，这一缺点还不太明显，对于长度为几万甚至几十万的词表，对于一个新的文本文档，向量化之后的长度都为词表的长度，因此最后生成的向量会很长，对内存需求会很大，最后就会降低算法效率。

&nbsp;&nbsp;&nbsp;&nbsp;我们可以使用哈希方法，将文本转化为数字。这种方法不需要词汇表，你可以使用任意长度的向量来表示，通过设定相对较小的长度可以大大减小计算资源，但这种方法不可逆，不能再转化成对应的单词。

### HashingVectorizer完整示例

- 使用sklearn中的HashingVectorizer来完成哈希向量化

```python
# 哈希向量化的极简方法，更复杂的用法参见API
from sklearn.feature_extraction.text import HashingVectorizer
corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
vectorizer = HashingVectorizer(n_features=8)
X = vectorizer.fit_transform(corpus)
print('HashingVectorizer result of corpus')
print(X.toarray())

test = ['This is a apple']
result = vectorizer.transform(test)
print('HashingVectorizer result of test')
print(result.toarray())
```

输出结果如下所示

    对于语料库中的每一个文本文档，Hash化之后的向量的维度都为我们预先设定的n_features的大小，即维度为8.

```python
HashingVectorizer result of corpus
array([[-0.89442719,  0.        ,  0.        ,  0.        ,  0.        ,
         0.4472136 ,  0.        ,  0.        ],
       [-0.81649658,  0.        ,  0.        ,  0.40824829,  0.        ,
         0.40824829,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        , -0.70710678,
         0.70710678,  0.        ,  0.        ],
       [-0.89442719,  0.        ,  0.        ,  0.        ,  0.        ,
         0.4472136 ,  0.        ,  0.        ]])

HashingVectorizer result of test
[[0.57735027 0.         0.         0.         0.         0.57735027
  0.57735027 0.        ]]
```


## LSA和LDA

### 主题模型概念

&nbsp;&nbsp;&nbsp;&nbsp;所有主题模型都基于相同的基本假设：
1. 每个文档包含多个主题；
2. 每个主题包含多个单词。

### LSA概念

&nbsp;&nbsp;&nbsp;&nbsp;潜在语义分析（LSA）是主题建模的基础技术之一。其核心思想是把我们所拥有的文档-术语矩阵分解成相互独立的文档-主题矩阵和主题-术语矩阵。

&nbsp;&nbsp;&nbsp;&nbsp;第一步是生成文档-术语矩阵。如果在词汇表中给出 m 个文档和 n 个单词，我们可以构造一个 m×n 的矩阵 A，其中每行代表一个文档，每列代表一个单词。在 LSA 的最简单版本中，每一个条目可以简单地是第 j 个单词在第 i 个文档中出现次数的原始计数。然而，在实际操作中，原始计数的效果不是很好，因为它们无法考虑文档中每个词的权重。例如，比起「test」来说，「nuclear」这个单词也许更能指出给定文章的主题。因此，LSA 模型通常用 tf-idf 得分代替文档-术语矩阵中的原始计数。

&nbsp;&nbsp;&nbsp;&nbsp;在实现Tf-idf时我们就发现了一个问题，tfidf的向量化表示非常稀疏，每一个document的向量表示都严重依赖于词表的长度，加入词表长度为100万，那么每一个document向量化表示后的长度度为100万，因此使用LSA或LDA这样的主题模型也可以实现降维的作用。

&nbsp;&nbsp;&nbsp;&nbsp;一旦拥有文档-术语矩阵 A，我们就可以开始思考潜在主题。问题在于：A 极有可能非常稀疏、噪声很大，并且在很多维度上非常冗余。因此，为了找出能够捕捉单词和文档关系的少数潜在主题，我们希望能降低矩阵 A 的维度。

&nbsp;&nbsp;&nbsp;&nbsp;这种降维可以使用截断 SVD 来执行。SVD，即奇异值分解，是线性代数中的一种技术。该技术将任意矩阵 M 分解为三个独立矩阵的乘积：M=USV，其中 S 是矩阵 M 奇异值的对角矩阵。很大程度上，截断 SVD 的降维方式是：选择奇异值中最大的 t 个数，且只保留矩阵 U 和 V 的前 t 列。在这种情况下，t 是一个超参数，我们可以根据想要查找的主题数量进行选择和调整。直观来说，截断 SVD 可以看作只保留我们变换空间中最重要的 t 维。

&nbsp;&nbsp;&nbsp;&nbsp;LSA 方法快速且高效，但它也有一些主要缺点：

- 缺乏可解释的嵌入（我们并不知道主题是什么，其成分可能积极或消极，这一点是随机的）
- 需要大量的文件和词汇来获得准确的结果
- 表征效率低

### LDA概念

&nbsp;&nbsp;&nbsp;&nbsp;LDA 即潜在狄利克雷分布，它使用狄利克雷先验来处理文档-主题和单词-主题分布，从而有助于更好地泛化。此处暂时不深入讨论LDA，只介绍一下sklearn中LDA的简要使用方法，更多LDA理念参阅 [Topic Modeling with LSA, PLSA, LDA & lda2Vec](https://medium.com/nanonets/topic-modeling-with-lsa-psla-lda-and-lda2vec-555ff65b0b05)这篇博客。

### 使用sklearn实现LSA和LDA

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

corpus = [
     'This is the first document.',
     'This document is the second document.',
     'And this is the third one.',
     'Is this the first document?']
# 创建Tf-idf模型
vectorizer = TfidfVectorizer()
# 使用给定的语料库来训练该模型
train_tfidf = vectorizer.fit_transform(corpus).toarray()

# 词表长度为9
print(len(vectorizer.vocabulary_)) # 9
print(vectorizer.vocabulary_) # {'this': 8, 'is': 3, 'the': 6, 'first': 2, 'document': 1, 'second': 5, 'and': 0, 'third': 7, 'one': 4}

lsa = TruncatedSVD(n_components=3)
# 使用tfidf特征来训练lsa模型
train_lsa = lsa.fit_transform(train_tfidf)

# lda = LatentDirichletAllocation(n_components=200)
# train_lda = lda.fit_transform(train_tfidf)

# 测试一个新的文本的主题分布
test = ['I really like this apple']
test_tfidf = vectorizer.transform(text)
print(lsa.transform(test_tfidf)) # [[ 0.4033972   0.09513381 -0.04013985]]

```

&nbsp;&nbsp;&nbsp;&nbsp;主题建模得到的结果中的每一个维度的取值代表了每一个主题的可能性大小。lsa得到的结果也相当于起到了降维的作用，在很多nlp任务中，可先用主题模型来构建特征，例如文本分类任务，可先用主题模型，进而使用svm等机器学习算法。


    train_tfidf，即tfidf表示的corpus，每个document的维度为9
    array([[0.        , 0.46941728, 0.61722732, 0.3645444 , 0.        ,
        0.        , 0.3645444 , 0.        , 0.3645444 ],
       [0.        , 0.65782665, 0.        , 0.25543054, 0.        ,
        0.60953246, 0.25543054, 0.        , 0.25543054],
       [0.53248519, 0.        , 0.        , 0.22314313, 0.53248519,
        0.        , 0.22314313, 0.53248519, 0.22314313],
       [0.        , 0.46941728, 0.61722732, 0.3645444 , 0.        ,
        0.        , 0.3645444 , 0.        , 0.3645444 ]])
    
    train_lsa, 即lsa后的corpus表示，每个document的维度为3，也相当于起到了降维的作用
    array([[ 0.95905678, -0.13453834, -0.24921784],
       [ 0.79765181, -0.18548718,  0.57388684],
       [ 0.45705072,  0.88833467,  0.04434133],
       [ 0.95905678, -0.13453834, -0.24921784]])

## Doc2vec向量化方法

### 概念
&nbsp;&nbsp;&nbsp;&nbsp;Doc2vec的原理与Word2vec非常类似，此处只就如何使用做一个简单介绍。

&nbsp;&nbsp;&nbsp;&nbsp;相比于Word2vec，Doc2vec在训练过程中会额外训练一个document的向量表示，所以在用gensim实现doc2vec时，需要先使用TaggedDocument对每个文档做一个tag标记。以gensim.test.utils. common_texts提供的数据集为例，原始数据格式如下：
    
    [['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']]
    
&nbsp;&nbsp;&nbsp;&nbsp;使用documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]对每个文档赋予tag之后格式如下所示：

    [TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]),
     TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]),
     TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]),
     TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]),
     TaggedDocument(words=['user', 'response', 'time'], tags=[4]),
     TaggedDocument(words=['trees'], tags=[5]),
     TaggedDocument(words=['graph', 'trees'], tags=[6]),
     TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]),
     TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]

### 使用gensim实现doc2vec的简单示例

```python
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import numpy as np

documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
# 定义并训练模型
model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
# 长度等同于用于训练的document的长度
print(len(model.docvecs)) # 9
# 查看模型训练后common_texts中第一句话的向量化形式
print(model.docvecs[0])  # array([ 0.00126504,  0.04400945,  0.00036975,  0.08506153, -0.00948037],dtype=float32)
# 与word2vec中用法一样，使用wv查看某一单词对应的词向量
word_vectors = model.wv['human']
print(word_vectors) # [ 0.09280109 -0.05563087 -0.0809787   0.08283566  0.03436603]
#  使用infer_vector得到一个新的文档的向量化表示
vector = model.infer_vector(["system", "response"])
print(vector) # [ 0.04092169  0.02866518  0.01339583 -0.05545134  0.06274231]
# 有多重方式将训练好的向量表达持久化，一种非常简单的方法：
docvecs = model.docvecs
x_train = []
for i in range(0, len(common_texts)):
    x_train.append(docvecs[i])
x_train = np.array(x_train)
```
