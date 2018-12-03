---
title: 机器翻译与Attention机制
description: Attention机制在NLP中最早应用于机器翻译中，本文以基于神经网络的机器翻译为例，介绍Seq2Seq模型和Attention机制。

categories:
 - NLP
tags: NLP Python
---

# 神经网络机器翻译
&nbsp;&nbsp;&nbsp;&nbsp;机器翻译是利用计算机将一种自然语言(源语言)转换为另一种自然语言(目标语言)的过程。近几年随着深度学习的兴起，使用神经网络的翻译系统的性能相比基于短语的翻译系统有了相当大的提升，传统的神经网络机器翻译模型为一个典型的Seq2Seq结构。
![](https://ws4.sinaimg.cn/large/006tNbRwgy1fxsrlyq8hwj31gt0u0h9z.jpg)

## Seq2Seq
&nbsp;&nbsp;&nbsp;&nbsp;Seq2Seq由编码器encoder和解码器decoder构成。为了缓解长依赖问题，encoder和decoder中的神经单元一般使用LSTM或GRU。对于翻译系统，在encoder阶段，依次读取源语言，并在最后得到一个表征所有输入文本的语义表征向量，并将该向量作为解码器decoder的输入。即使用了编码器的最后一个隐层状态作为解码器的首个状态。

&nbsp;&nbsp;&nbsp;&nbsp;对于下图中的例子，下半部分表示encoder，C为经过encoder后得到的语义表征向量，并将C输入给图中上半部分中的decoder模块。

![](http://cloudpicture.oss-cn-hangzhou.aliyuncs.com/18-12-3/98298919.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;可将decoder看作条件递归语言模型，在计算decoder的节点时使用的是条件概率模型。

### 传统Seq2Seq模型的缺点
&nbsp;&nbsp;&nbsp;&nbsp;传统的Seq2Seq模型只是利用了encoder 最后一个state作为decoder的初始state。将输入语句的所有信息都保存在了encoder中的最后一个state中，在解码过程中容易遗漏信息，虽然在decoder中使用LSTM这种门限模型可以在一定程度上缓解这一问题。但是需要尽可能多的在decoder的传播过程中携带更多的信息，才能在生成目标语句的时候，仍然能够保留源语句的语义信息。

&nbsp;&nbsp;&nbsp;&nbsp;针对上述缺点，可以做一个小的改进，将encoder最后一层喂给decoder的每一层。从某种意义上来说，在decoder的每个时刻都能访问到输入。(与Attention的思想有些类似，都是在decoder的每一个阶段都利用了encoder阶段的信息).该方法如下图所示。
![](https://ws4.sinaimg.cn/large/006tNbRwgy1fxsrweyfgxj317m0fugty.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;相比于上述中的的改进方法，Attention机制能够更好的解决此问题。


## Attention机制
### 概念
![](http://cloudpicture.oss-cn-hangzhou.aliyuncs.com/18-12-3/60316289.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;先来看看什么是Attention机制，对于'She is eating a green apple'这句话，当我们看到 “eating” 这个词的时候，我们的注意力会马上集中在某种食物的名字上，而对其他形容词则相应的注意力会降低。 

&nbsp;&nbsp;&nbsp;&nbsp;利用注意力机制，不是简单的使用encoder最后一个节点状态表示全部语义，而是用到编码器encoder的所有隐层状态信息。

&nbsp;&nbsp;&nbsp;&nbsp;通俗的说，就是在decoder阶段得到每一个时间步的输出结果时，同时关注encoder的每一个隐层状态信息，但是不同的encoder隐层状态的重要性不同，所以我们需要给encoder的每一个隐层状态赋予一个不同的权重，这个权重即注意力权重。

###  具体过程
&nbsp;&nbsp;&nbsp;&nbsp;那么如何表示encoder中不同隐层状态的重要性呢？也就是最简单的方法，给每个节点打个分，评分的过程为：以前一刻的decoder状态(h<sub>t-1</sub>)和每个encoder状态(h<sub>s</sub>)为参数，作为注意力机制的判别基准，输出得分。以下图为例，蓝色表示编码器encoder，红色表示解码器decoder。在计算问号表示的结果时，首先以问号前一节点(椭圆标注的)依次和encoder中的每一个隐层状态(椭圆标注的蓝色方框)为参数，计算score评分函数。

![](https://ws4.sinaimg.cn/large/006tNbRwgy1fxsultkw6lj30vy0pwwhx.jpg)
- 蓝色：编码器encoder
- 红色：解码器decoder

> 图中第二层代表隐层状态h。

&nbsp;&nbsp;&nbsp;&nbsp;得分函数会给encoder编码器的每个隐层生成一个分数，可理解为代表分配了多少注意力。

![](https://ws2.sinaimg.cn/large/006tNbRwgy1fxsun8e1izj30y80k6dhu.jpg)
&nbsp;&nbsp;&nbsp;&nbsp;对encoder的所有隐层计算出相应的score分数后，通过softmax计算出对应的注意力权重，如上图所示。

![](https://ws3.sinaimg.cn/large/006tNbRwgy1fxsuq2j6xrj31bw0r8djz.jpg)

&nbsp;&nbsp;&nbsp;&nbsp;因为不同时刻，计算出的权重值不同， 表示对encoder的不同位置分配的注意力不同。之后将编码器encoder的所有隐层状态hs，根据注意力机制给出的权重，通过求加权和将其组合起来。这个加权和即上下文向量 context vector。

&nbsp;&nbsp;&nbsp;&nbsp;接下来利用上下文向量，来生成decoder中的下一个单词

### 评分函数(注意力函数)常用的几种选择：
![](https://ws3.sinaimg.cn/large/006tNbRwgy1fxsu60opjyj311q0hq42p.jpg)
1. 使用解码器隐层状态和编码器隐层状态的内积。相当于找相似的词，即找出相同语义的词，并基于此生成分数。
2. 双线性注意力函数。相比方法1中的点乘，此方法有一个矩阵W，中间矩阵W相当于学习到如何将不同权重分配到点乘的不同部分。
3. 使用了单层神经网络，做一次矩阵乘法再通过一次tanh函数变换，其中将解码器和编码器的隐层状态进行了拼接，v和W都是神经网络学习到的参数。

### 其他注意力模型
> 以下内容简单了解即可，一般不会使用

1. 局部注意力模型

&nbsp;&nbsp;&nbsp;&nbsp;每次只将注意力放到一部分状态上，类比从记忆中检索某些事物的概念
![](https://ws4.sinaimg.cn/large/006tNbRwgy1fxsui3ijqdj30uc0jk76y.jpg)

2. Double Attention(双重注意力机制)

&nbsp;&nbsp;&nbsp;&nbsp;在机器翻译中同时对源语言和目标语言进行注意力机制。

3. fertility

&nbsp;&nbsp;&nbsp;&nbsp;和Double Attention思想相反，该想法认为将注意力机制放在相同的位置会带来负面的影响，例如在机器翻译中，有时候一单词会被翻译成目标语言中的多个单词。即会造成翻译得到的目标语言中出现多个重复的单词。

## 机器翻译中decoder解码方式
&nbsp;&nbsp;&nbsp;&nbsp;在机器翻译中，即使已经有了Seq2Seq结构，Attention机制，还有一个问题需要考虑，那就是在decoder阶段如何得到目标语言，主要有如下几种方式：

1. 连续采样
生成一个词后，基于目前所得的概率分布，对下一个词进行采样，并重复此过程。缺点：方差很大，每次解码同一个句子也经常得到不同的结果。

2. 贪婪搜索
每一步都是在给定前面序列的情况下选出最好的词。缺点：没有留有犯错空间，比如第一个词翻译的就不太好时，以后的结果可能将一直得到较差的结果。

3. 束搜索(beam search)
相当于留有犯错空间的贪婪搜索。

&nbsp;&nbsp;&nbsp;&nbsp;在基于上述所介绍的神经网络机器翻译中，目前最好的方法是小尺寸的beam search区间，例如5到10, 而基于短语的翻译系统一般需要较大的beam search区间，例如100到150.
