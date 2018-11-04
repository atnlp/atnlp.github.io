---
title: 文本分类中的Attention注意力机制及PyTorch实现
description: 不同于机器翻译的注意力机制，文本分类中的注意力机制没有目标语言这一概念，需要引入自注意力机制(self-attention)的概念, 本文以Hierarchical Attention Networks for Document Classification这篇paper为例进行介绍，进而实现一个带有Attention机制的双向GRU模型。
categories:
 - PyTorch
tags:
---

# Attention

## 概念

&nbsp;&nbsp;&nbsp;&nbsp;NLP领域中的Attention机制最早开始在机器翻译中使用，
以NLP中的中英机器翻译为例，思考一下人工翻译的过程，人工翻译并不会通过读整个中文句子，然后从零开始， 一次就完整的翻译成一个英语句子。人工在翻译时首先会做的可能是先翻译出句子的某一部分， 看下一部分，并翻译这一部分。以此反复从而翻译完一整个句子。在这个中英翻译的过程中，对中文语句的每一小部分甚至每一个不同的词汇都会有不同的侧重点，这也就是Attention的含义，对于一条语句中的不同部分赋予不同的注意力(重视度)。

&nbsp;&nbsp;&nbsp;&nbsp;传统上的Attention的应用，总是要求我们的任务本身同时有源和目标的概念。比如在机器翻译里, 我们有源语言和目标语言，以中英翻译为例，源语言即中文，目标语言即英文。一句中文包含很多单词，而这些单词中可能某几个单词对于翻译结果具有极重要的作用，而有一些单词对最终的翻译结果没有多大的作用。Attention即赋予源语言中的词汇不同的权重。

&nbsp;&nbsp;&nbsp;&nbsp;但是还有很多任务不同时具有源和目标的概念。比如文本分类,它只有text文本，没有目标语言/文章， 这就需要一个变种的技术，自注意力机制（self-attention或者intra-attention）, 顾名思义，就是原文自己内部的注意力机制。

&nbsp;&nbsp;&nbsp;&nbsp;[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)这篇文章提出的应用于文本分类的Attention具有极广的影响力，核心思想可用下图概括：
        
![](http://cloudpicture.oss-cn-hangzhou.aliyuncs.com/18-11-4/75717492.jpg)
这篇paper从句子和词汇两个角度来对文本分类任务进行考量。
- 在这篇文章中，哪些句子更重要，能够决定它的分类？ 
- 在这篇文章的某个句子中，哪些词语最重要，能够影响句子在文章里的重要性？

&nbsp;&nbsp;&nbsp;&nbsp;这篇文章引入了context vector的概念，因为对于文本分类任务，缺少了与自动翻译中相类似的目标语言的概念，可以把context vector理解成self-attention中的目标语言。Uw即词语级别的context vector, Us即句子级别的context vector. Uw和Us手动定义，并经神经网络训练来不断迭代。

&nbsp;&nbsp;&nbsp;&nbsp;以词语级别的Attention为例(即上图中的下半部分)，核心公式如下图所示：
![](http://cloudpicture.oss-cn-hangzhou.aliyuncs.com/18-11-4/22873750.jpg)
- 公式一中，Ww 与bw为Attention的权重与bias. bias项可根据具体情况选择是否保留，Uw即词级别的context vector
- 公式二即对Ut和Uw的softmax过程
- 公式三级为增加Attention后，对于输出结果各个时间步中的权值大小

## 代码示例

### 双向GRU + Attention模型定义

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_dim = 256
embedding_dim = 300
vocab_size = 10000

class bigru_attention(BasicModule):
    def __init__(self):
        super(bigru_attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru_layers = 2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双向GRU，//操作为了与后面的Attention操作维度匹配，hidden_dim要取偶数！
        self.bigru = nn.GRU(embedding_dim, hidden_dim // 2, num_layers=gru_layers, bidirectional=True)
        # 由nn.Parameter定义的变量都为requires_grad=True状态
        self.weight_W = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.weight_proj = nn.Parameter(torch.Tensor(hidden_dim, 1))
        # 二分类
        self.fc = nn.Linear(hidden_dim, 2)

        nn.init.uniform_(self.weight_W, -0.1, 0.1)
        nn.init.uniform_(self.weight_proj, -0.1, 0.1)

    def forward(self, sentence):
        embeds = self.embedding(sentence) # [seq_len, bs, emb_dim]
        gru_out, _ = self.bigru(embeds) # [seq_len, bs, hid_dim]
        x = gru_out.permute(1, 0, 2)
        # # # Attention过程，与上图中三个公式对应
        u = torch.tanh(torch.matmul(x, self.weight_W))
        att = torch.matmul(u, self.weight_proj)
        att_score = F.softmax(att, dim=1)
        scored_x = x * att_score
        # # # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1)
        y = self.fc(feat)
        return y
```