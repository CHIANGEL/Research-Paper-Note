# Research-Paper-Note
My research paper notes, focusing on data mining/recommender/reinforcement learning. This README.md file is a total collection of all papers in chronological order (key words provided for indexing). There are subcatalog markdown files available for area-specific indexing.

我的论文笔记，主要聚焦于数据挖掘、推荐系统、强化学习。这里是所有论文笔记的汇总，按照时间顺序排列（提供了关键词方便检索），同时可以去具体的分类markdown文档中检索论文。

## Click Models for Web Search

链接：[Survey: Click Models for Web Search](https://www.semanticscholar.org/paper/Click-Models-for-Web-Search-Authors-%E2%80%99-version-*-Chuklin-Markov/0b19b37da5e438e6355418c726469f6a00473dc3?p2df)

关键词：Click Model, CM, Probabilistic Graphical Models, PGM

这篇15年的综述对PGM时代（前深度学习时代）的Click Models进行了完整的总结，基本之后我们做Click Model的文章必然会引用，好文推荐。

## A Neural Click Model for Web Search

链接：[https://dl.acm.org/doi/10.1145/2872427.2883033](https://dl.acm.org/doi/10.1145/2872427.2883033)

关键词：NCM, Click Model, LSTM, User Modeling

以前的click model都是基于概率图模型的，这篇WWW16的文章第一次用神经网络和深度学习去建模click model任务，算是开创性的工作。模型本身十分简单，就是用一个RNN去表示一个doc list的sequence process。文章中没有给出实现代码，我自己成功复现了一版代码[[Link](https://github.com/CHIANGEL/Neural-Click-Model)]，需要注意，我的复现版本和原论文有几点不一样的地方（个人认为是改进）：

- 我没有采用原论文中one-hot的embedding方法，而是直接用PyTorch的nn.embedding对每一个id进行feature初始化，然后让embedding参与梯度更新。原论文的one-hot方法非常笨重，一个embedding可以长达10240维，不仅稀疏，而且很难放入GPU加速。
- 我额外加入了vertical type信息，这个取决于我们采用的数据集有没有这些额外信息。
- 我在每一个RNN的time step都输入了query信息，而不只是用query信息去做初始化，防止query信息被遗忘。

## A Context-Aware Click Model for Web Search

链接：[https://dl.acm.org/doi/10.1145/3336191.3371819](https://dl.acm.org/doi/10.1145/3336191.3371819)

关键词：CACM, Click Model, User Modeling

对Click Model任务进行更加复杂的建模，参考PGM时代Click Model的Examination Hypothesis，CACM将Click Probability分成Examination Prediction和Relevance Estimation两部分建模，然后通过一个Combination Layer进行结合计算出每一个doc的点击概率。可以理解为是通过堆积复杂模型来达到更好的性能，个人感觉没有NCM优雅和灵活。

最重要的是，我认为CACM中的Relevance Estimator的建模是存在一定的问题的，从下图看，Relevance Estimator的Click Context Encoder中已经将document的序列信息也建模进来！但是常识中，query-doc的相关性分数relevance score应该是一个和doc位置信息无关的东西，CACM的建模方式却意味着，**一模一样的query-doc pair，一旦doc的位置发生改变，这个query-doc pair的relevance score也就会发生改变**，这是一件不合理的事情。

![CACM-FIG1](./images/CACM-FIG1.JPG)

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

## Graph Attention Networks

链接：[https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

关键词：Attention, GAT, Graph

ICLR2018的工作，比GraphSage迟一年。GAT将Attention机制引入图神经网络，但和GraphSage类似，还是每次通过将目标结点的一阶邻居想自己汇总信息，来得到最新的node embedding，只不过这个“汇聚”过程不再是简单取平均，而是通过attention机制的加权平均，而attention系数则是有一个**全结点共享**的线性变化层求得：

![GAT-eq1](./images/GAT-eq1.JPG)

公式中，LeakyReLU及其里面的a就是那个全节点共享的线性变化层，input是目标结点i和他的一个一阶邻居j的embedding。LeakyReLU外面套的就是一个softmax，即可得到目标结点i的所有一阶邻居j的attention系数（i也算自己的一阶邻居，即添加自环self-loop）。另外，值得注意的是，上面这个attention系数计算公式是对两个节点i,j对称的，因此其实就是图中每条边可以计算出一个attention系数。接着通过对所有一阶邻居的加权得到目标结点i的新embedding：

![GAT-eq2](./images/GAT-eq2.JPG)

同时，文章还提出留用multi-head attention来稳定GAT的学习过程，也就是简单的独立的计算多组GAT，然后通过取平均或者拼接的方式进行集成。

![GAT-EQ3](./images/GAT-EQ3.JPG)

![GAT-EQ4](./images/GAT-EQ4.JPG)

## Inductive Representation Learning on Large Graphs

链接：[https://arxiv.org/abs/1706.02216](https://arxiv.org/abs/1706.02216)

关键词：GNN, GraphSage

GraphSage应该是第一个提出图神经网络中inductive和transductive区别的工作（不太确定），transductive是指test set中的节点必须在training set中出现过（只是没有标签），而inductive则是指test set中的节点不一定出现在training set中，也就是在测试阶段允许出现全新的节点，那么难点就在于这些出现的全新节点其实并没有一个很好的embedding（因为没有参与训练），所以inductive比transductive更具备挑战性，也更贴近实际应用场景。

GraphSage通过对目标结点的邻居信息进行聚合操作，来生成目标节点的embedding（可以给目标节点添加自环，使之也考虑自身信息），所以GraphSage主要分为两步，首先是对目标节点的周围邻居进行聚合aggregation，得到一个新的embedding，然后将这个新embedding和目标节点原本的embedding进行拼接、过一个全连接层，最终得到目标节点的output embedding。

![GraphSage-FIG1](./images/GraphSage-FIG1.JPG)

上述是GraphSage的通识，然后有三点可以变通和思考的地方：

- 聚合aggregation有多种不同的函数可以选择，比如简单取平均、pooling、LSTM等等，本质上就是把多个input vector信息融合成一个vector
- 一次aggregation没必要把目标节点的所有邻居都拿来算，我们可以采用Neighbor Sampling的技术，每次只从目标节点的邻居集合中采样出K个即可（集合大小不足K，那就有放回的采样），这样既可以使得计算时的数据维度对齐，方便GPU加速，同时也是一种assemble方法。
- 对目标节点，我们可以多次aggregation，每一次聚合，就是向外扩展一层邻居，如此就可以聚合高阶邻居的信息

## Heterogeneous Graph Attention Network

链接：[https://arxiv.org/abs/1903.07293](https://arxiv.org/abs/1903.07293)

关键词：HGAT, HAN, Heterogeneous

原论文给这个模型起名缩写是HAN，但因为该模型可以理解为是GAT在异质图上的扩展，因此有不少人称之为HGAT。本文的attention机制可以分类两层，第一层是node-level attention，第二层是semantic-level attention（其实就是meta-path-level attention），但这里的attention系数是用MLP或者向量内积算出来的。

文章首先给出了一系列的定义，包括meta-path、meta-path based neighbor之类的，详见原文。假设数据集有N个节点，人为定义P类meta-path，对每一个结点i，我们通过node-level attention的到了结点i的P个embedding，此时一共N*P个embedding。

![HGAT-FIG1](./images/HGAT-FIG1.JPG)

接着，在semantic-level attention阶段，我们要转换思维。在此之前，这N*P个embedding我们会潜意识中将它们分为N类（即按照结点分），接下来我们要按照meta-path将这些embedding分为P类，然后通过semantic-level attention计算出P个attention系数，然后将P类合成一类，得到最终的N个embedding。注意，下图中的取平均，就是对每一类meta-path的N个embedding算出的coefficient求平均，因为当时看论文的时候困惑了一下，所以这里我特意讲到“转换分类思维”的想法。

![HGAT-FIG2](./images/HGAT-FIG2.JPG)

## TGCN: Tag Graph Convolutional Network for Tag-Aware Recommendation

链接：[https://dl.acm.org/doi/10.1145/3340531.3411927](https://dl.acm.org/doi/10.1145/3340531.3411927)

关键词：Tag-Aware, TGCN, Graph, TRS

TRS (Tag-Aware Recommender)，即user在和item产生交互的同时，会主动给item打上一些标签tag，那么这些tag不仅会反映item的属性，还会反映user的潜在兴趣，是可以利用的side information。本文提出了三点motivation（其实和其他用图做推荐的都大同小异）：

- Cold Start，老生常谈的问题，GNN论文必备motivation
- Ambiguity，tag一般是用字符串表示，而一样的tag对不同的item和user会有不同的含义，比如apple可以是水果，也可以是手机
- Redundancy，和ambiguity对应，不同的tag可能表示一样的含义，比如mobile phone和ceil phone

本质上Ambiguity和Redundancy都是因为单纯的tag缺乏上下文和语义信息，如果引入图，tag节点的邻居就可以提供辨别信息，解决这些问题。

首先是构图，文章的构图方法很简单，一次user和item的交互会产生若干三元组(user, item, tag)，对每一个三元组，我们就分别将user、item、tag彼此连边，同时user-item边权重为1，user-tag和item-tag的权重取决于这个tag在user/item邻居中出现的次数，也就是说，tag出的次数越多，说明这个tag对user/item越重要。

![TGCN-FIG1](./images/TGCN-FIG1.JPG)

TGCN比较特别的地方在于它做了两层aggregation（node-level和type-level，HGAT/HAN有异曲同工之妙）。对于一个中心节点，在node-level aggregation中，我们先对它的邻居按类型进行分别sample和attention，在本文中，也就是三类邻居节点，分别求出三个embedding用以表示中心节点的邻居信息，然后在通过type-level aggregator进行邻居之间的信息整合。

显而易见，我们在type-level aggregation也可以简单地采用attention机制，但是TGCN认为我们应该将特征交叉的思想拿进来，所以TGCN没有把所有的node-level邻居向量用attention合成一个，而是拼成一个2D矩阵（类似于图片），然后参考CV中的卷积层，提出了横向/纵向两类卷积特征交叉方法，具体可以看论文。

type-level aggregation结束后我们就得到了user和item的embedding，最后一个简单的内积就可以求得user-item的CTR。

除此之外，TGCN还提出了TransTag，其实就是在端到端训练的loss function中加入一项。具体来说，就是参考了TransE的思路，认为tag是连接user和item的一类特殊边，所以在embedding层面上，user + tag应该和item尽量接近。这一点的提出也有很形象的可解释性，user1和user2都和item1发生了交互，但他们发生交互的原因可能是不一样的，比如user1是因为价格低，user2是因为好看，而这样的区别就是由tag提供的。