# MobileBERT paddle

使用 `paddlepaddle` 实现 `transformers` 中提供的 `MobileBERT` 模型。


论文： **MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**

https://arxiv.org/abs/2004.02984


huggingface 模型页面：  

https://huggingface.co/google/mobilebert-uncased


transformers 源代码：

https://github.com/huggingface/transformers/tree/master/src/transformers/models/mobilebert


## 论文和代码解析

本文对bert模型进行了知识迁移，把大模型的知识迁移到小模型中。

深层网络会很难训练，尤其是在小模型中我们把模型的“腰”收的很紧，这样就更不容易训练了。所以这里作者采取的方法是，先训练一个大尺寸的网络作为教师，然后在小模型（学生）网络的设计中，把每一层的 feature map 设计成相同的形状。这样，就可以在训练时，让这两个模型尽量对齐。

### 模型设计

这里结合论文和代码，对模型设计进行介绍。我想用一个先总后分的方法来讲。

先自上而下，讲解模型的构造，再从下到上，看每一个子网络的实现。

```python
class MobileBertModel(MobileBertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings = MobileBertEmbeddings(config)
        self.encoder = MobileBertEncoder(config)
        self.pooler = MobileBertPooler(config) if add_pooling_layer else None
```

`MobileBertModel` 包括 embedding 和 encoder，以及一个 pooler。（原文中没有 pooler ）





### 训练策略












