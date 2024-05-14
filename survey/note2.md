## pFedGraph
[论文](https://proceedings.mlr.press/v202/ye23b.html "Personalized Federated Learning with Inferred Collaboration Graphs")
[项目](https://github.com/MediaBrain-SJTU/pFedGraph?tab=readme-ov-file)
1. 关键模块
   - 基于服务器上的成对模型相似性和数据集大小推断合作图，以促进细粒度合作（服务器端）
   - 在客户端优化本地模型，通过聚合模型的协助来促进个性化（客户端）
2. 核心思想：学习一个**合作图**，该图模拟了每对合作的收益，并分配适当的合作强度。

服务器通过促进模型相似性的合作来学习合作图，然后根据学习到的合作图为每个客户端获得聚合模型。在客户端，通过平衡经验任务驱动的损失和本地个性化模型与从服务器发送的聚合模型之间的相似性来优化个性化模型。
## FedETF
[论文](https://arxiv.org/abs/2303.10058 "No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier")
[项目](https://github.com/ZexiLee/ICCV-2023-FedETF)

受到**神经崩溃**（Neural Collapse）现象的启发，通过在训练过程中使用合成的和固定的简单多面体等角紧框架（Simplex Equiangular Tight Frame，简称**ETF**）分类器来解决分类器偏差问题。

利用一个最优的分类器结构，使得所有客户端能够在极端异质的数据环境下学习到统一和最优的特征表示。

> **神经崩溃现象** : 在完美训练场景下，特征原型和分类器向量会收敛到一个最优的简单多面体等角紧框架（ETF）

在FL训练期间使用合成的和固定的ETF分类器。这种最优的分类器结构使得所有客户端能够学习到统一和最优的特征表示，即使在极端异质的数据环境下也是如此。

还提出了一种局部微调策略，用于在FL训练后提高每个客户端的个性化水平。通过这种策略，客户端可以根据自己的本地数据分布调整全局模型，从而实现更好的个性化。
## FedCP
[论文](https://dl.acm.org/doi/abs/10.1145/3580305.3599345 "FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy")
[项目](https://github.com/TsingZ0/FedCP)

通过为每个样本生成一个条件策略来分离特征中的**全局信息**和**个性化信息**，并通过**全局头部**和**个性化头部**分别处理这些信息。

1. 分离特征信息:
FedCP的关键组件是**条件策略网络**（Conditional Policy Network，**CPN**），它负责生成样本特定的策略来分离特征中的全局和个性化信息。CPN由一个全连接层（FC层）和层归一化层（Layer Normalization）组成，后面跟着ReLU激活函数。CPN的输入是样本特定的向量和客户端特定的向量，这两个向量通过Hadamard乘积（元素相乘）结合在一起，形成CPN的输入。

    在每个客户端上，FedCP通过以下步骤分离特征信息：

   - 使用全局特征提取器生成特征向量。
   - 通过CPN生成样本特定的策略。
   - 将策略应用到特征向量上，分离出全局特征信息和个性化特征信息。

2. 处理特征信息:
分离出的特征信息被送入两个不同的路径：全局头部和个性化头部。全局头部冻结了全局信息，而个性化头部则在本地进行训练以捕捉个性化信息。最终的输出是全局头部和个性化头部输出的加权平均，其中权重由样本特定的策略决定。

3. 对齐特征:
为了使个性化特征提取器的输出与全局头部的特征相匹配，FedCP使用最大均值差异（Maximum Mean Discrepancy，MMD）损失来对齐这两个特征表示。这有助于在保持全局一致性的同时，允许客户端学习到个性化的特征表示。

## pFedPG
[论文](https://arxiv.org/abs/2308.15367 "Efficient Model Personalization in Federated Learning via Client-Specific Prompt Generation")

核心思想: 提出了一个名为**客户端特定提示生成**（pFedPG）的新型个性化FL框架。这个框架在服务器上学习部署一个个性化的提示生成器，用于为每个客户端生成特定的视觉提示，从而高效地将冻结的模型主干（backbone）适配到本地数据分布。

pFedPG框架通过交替进行个性化提示适配和个性化提示生成两个阶段来实现高效的模型个性化:
- **本地的个性化提示适配** 在客户端，pFedPG利用可训练的参数（提示）来适应大规模预训练模型。这些提示被视为客户端特定的可学习参数，并通过梯度下降在训练过程中直接优化。通过这种方式，模型能够高效地适应客户端的数据分布，同时避免了更新整个大型模型所带来的计算负担。
- **全局的个性化提示生成** 在服务器端，pFedPG学习一个个性化提示生成模块，该模块能够利用客户端的底层特征来生成个性化提示。服务器无法直接访问客户端的私有数据，因此它通过观察本地优化方向（即客户端训练后提示的变化）来隐式地获取客户端特性。这种方法允许服务器为每个客户端生成有助于本地适应的个性化提示。

## DBE
[论文](https://arxiv.org/abs/2311.14975 "Eliminating Domain Bias for Federated Learning in Representation Space")
[项目](https://github.com/TsingZ0/DBE)

DBE(Domain Bias Eliminator for federated learning)框架包含两个模块：

1. 个性化表示偏差记忆 (PRBM)
   - 由于统计异质性的存在，本地模型在接收到全局模型参数后，会倾向于学习有偏的表示。为了解决这个问题，作者提出了PRBM模块，它的作用是将表示偏差从原始表示中分离出来，并在每个客户端上本地保存。
   - 具体来说，原始的特征表示z被视作全局部分zg_i和个性化部分?zp_i的组合，即`z_i := zg_i + ?zp_i`。在训练过程中，特征提取器的输出被更改为zg_i，而?zp_i则在本地保持可训练。?zp_i是客户端特定的，但对于客户端上的所有本地数据都是相同的，因此它记住了客户端特定的均值。通过这种方式，特征提取器转向捕获更少偏差的特征信息zg_i。
2. 均值正则化 (MR)
   - 在没有明确指导的情况下，特征提取器很难自动区分表示中的有偏和无偏信息。为了使特征提取器专注于无偏信息，并进一步分离zg_i和?zp_i，作者提出了MR模块，它明确指导本地特征提取器生成具有客户端不变均值的zg_i。
   - MR通过将zg_i的均值正则化到共识的全局均值?zg来实现，这与?zp_i中记忆的客户端特定均值相反。具体来说，MR通过独立地在每个特征维度上将zg_i的均值正则化到?zg来实现。通过这种方式，特征提取器被鼓励专注于无偏信息，从而提高通用表示质量。
  
## GPFL
[论文](https://arxiv.org/abs/2403.17833 "GPFL: A Gradient Projection-Based Client Selection Framework for Efficient Federated Learning")
[项目](https://github.com/TsingZ0/GPFL)

GPFL(Global and Personalized Feature Learning)方法的核心是将神经网络的主干网络（backbone）分为**特征提取器**（φ）和**头部**（ψ）。特征提取器负责将输入数据映射到一个低维特征空间，而头部则将这些特征映射到类别空间以进行分类。

1. **特征提取**（Feature extraction）：
   使用特征提取器φ将数据样本映射到低维特征空间，得到特征向量fi。
2. **特征转换**（Feature transformation）：
   通过条件计算技术，将原始特征向量fi转换为两个特征向量fGi（全局特征）和fPi（个性化特征），分别用于全局和个性化任务。
3. **全局类别嵌入**（Global Category Embedding, GCE）：
   - 引入可训练的全局类别嵌入，通过GCE层获取类别嵌入，这些嵌入在全局层面上指导特征提取。
   - 使用角度级和幅度级全局指导损失（Lalg_i 和 Lmlg_i），确保特征向量与对应的类别嵌入在角度上接近，在幅度上保持一致。
4. **条件阀门**（Conditional Valve, CoV）：
   - CoV用于在客户端模型中创建全局指导路径和个性化任务路径，以便同时学习全局和个性化特征信息。
   - CoV根据全局条件输入g和个性化条件输入pi生成转换参数γ、β、γi和βi，这些参数用于特征转换过程。
5. **个性化任务**（Personalized tasks）：
   客户端i使用fPi学习头部ψ，将转换后的特征空间映射到类别空间，并通过交叉熵损失函数LP_i进行优化。
6. **局部损失函数**（Local loss function Li）：
   将上述损失函数组合起来，形成客户端i的总损失函数Li，包括个性化任务损失、角度级和幅度级全局指导损失，以及正则化项。
## ViTFL
1. ViT
2. ResNet