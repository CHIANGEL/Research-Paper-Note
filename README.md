# Research-Paper-Note
My research paper notes, focusing on data mining/recommender/reinforcement learning. 我的论文笔记，主要聚焦于数据挖掘、推荐系统、强化学习

## An End-to-End Neighborhood-based Interaction Model for Knowledge-enhanced Recommendation

链接：[https://arxiv.org/abs/1908.04032](https://arxiv.org/abs/1908.04032)

关键词：NI, KNI, Graph, Recommender, KDD Workshop Best Paper

本文主要解决三个问题：

- Data Sparsity
- Cold Start
- Early Summarization

前两个是RecSys中常见的问题，一般GNN的方法都可以缓解，第三个问题则是本文提出的一个新概念，意思是寻常GNN的方法是直接利用目标结点的邻居去生成embedding，这样并没有充分的利用图结构的local structure（比如邻居与邻居之间的交互），因此这个embedding是被“过早”的生成了，缺失了一部分有价值的local信息。

接着，文章以Average Aggregation和Attention Aggregation为例，总结了目前GNN的一种通式：

![KNI-eq1](./images/KNI-eq1.JPG)

![KNI-eq2](./images/KNI-eq2.JPG)

![KNI-eq3](./images/KNI-eq3.JPG)

![KNI-eq4](./images/KNI-eq4.JPG)

其中矩阵A是一个系数矩阵，侧重于local structure；矩阵Z则是一对目标结点u，v的邻居的两两组合的内积矩阵，侧重于全局信息。而之前提到的Average Aggregation和Attention Aggregation，其实就是这个通式的特例，即改变矩阵A。因此我们也可以进一步对A进行改进，加入“邻居和邻居”之间的互动关系，从而得到NI模型的公式：

![KNI-eq1](./images/KNI-eq5.JPG)

其实相对于Attention Aggregation，NI的不同就在于拼接了更多的结点信息，从而更好地利用邻居信息，一定程度上解决Early Summarization的问题。

上面的讨论还是仅限于User-Item的图结构，如果我们通过引入知识图谱的信息，即更多的Entity和Relation，就可以构建KNI。相比于NI，KNI的区别就是图结构更加丰富，公式和计算过程没有改变。除此之外，KNI和NI都运用了Neighbor Sampling的技术。而也正是因为采用了NS技术，KNI/NI在Evaluation阶段会有一个类似assemble的过程，eval次数越多，sample次数越多，性能会有进一步的提升。因此，文章是每次进行40次eval，然后取平均。

## User Behavior Retrieval for Click-Through Rate Prediction

链接：[https://arxiv.org/abs/2005.14171](https://arxiv.org/abs/2005.14171)

关键词：UBR4CTR, User Behavior Retrieval

CTR预测一般会用到用户的历史信息来产生personalized的预测结果，比如DIN、DIEN都是这方面经典的工作。但是如果用户的历史信息序列过长，我们不可能把该用户所有的历史信息都一股脑放入模型进行预测，这样还带来两方面的坏处：

- 运算时间过长
- 过长的历史信息序列会包含很多的无用信息和噪声，反而不利于预测

因此，目前学界、工业界常用的操作就是直接做一个截断，只把该用户最近的历史信息序列截取过来做CTR预测，这样的操作简单，但是可能会遗漏有价值的信息，比如用户的长期、周期性的行为规律。因此如果从用户特长的历史信息序列中筛选出有价值、低噪声的历史信息用于CTR预测是很重要的，本文就提出了一种基于Attention的用户历史信息筛选机制，来取代目前“简单截断历史信息序列”的粗暴方法。

![UBR4CTR-fig1](./images/UBR4CTR-fig1.JPG)

这其实是一种很简单但很重要的思想，即应该筛选用户的历史信息而不是简单截断。至于这个“筛选”的方法，就可以千变万化，这篇文章中提到的只是一种可能的筛选机制，比如在工业界部署上，为了简化，我们可以通过制定规则来实现筛选，比如和target item属于同一类别的item历史信息应当被重点考虑。