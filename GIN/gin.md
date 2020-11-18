### GIN

关键思想：

使用领域聚合的GNN模型无法优于WL test，后者是GNN分类能力的上限

要设计一个达到上限的GNN分类模型，需要满足：

- 层与层之间的映射应该是injective
- 最后输出时应该设计多层感知机
- feature应该是可数的，使用sum的方法聚合特征，同时聚合函数必须是injective的（比如文中使用的类似one-hot encoding的方式）

在pytorch-geomatric下，这种模型的层间关系为：

![image-20201118140932527](../PIC/image-20201118140932527.png)

最后，GIN模型适合处理节点特征为1维（取值还不是连续的，是离散的）的图，因为高维度的图要达到feature可数实在是非常困难，因为feature可数意味着feature必须是离散的，并且只能取到确定的值