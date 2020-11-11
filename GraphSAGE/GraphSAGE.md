## GraphSAGE

逐层学习，每层的同节点理论上参考的邻居结点（所有或者部分）应该保持一致

学习一个通过邻居节点与自身feature input生成feature output的算法



### 与GCN的关系

正常GraphSAGE的层生成函数

![image-20201111154125726](/Users/luckytiger/Library/Application Support/typora-user-images/image-20201111154125726.png)

如果使用Mean Aggregator：

![image-20201111154206079](/Users/luckytiger/Library/Application Support/typora-user-images/image-20201111154206079.png)



考虑GCN的公式
$$
Z = \widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}X\Theta
$$
可以写成
$$
Z = \hat AX\Theta
$$
$\hat A$是邻接矩阵加上自身环之后的归一化结果，性质与邻接矩阵应该没有大差别

所以用这个矩阵乘以input feature矩阵X，可以认为每个节点的属性，被更新成了它所有相邻节点和自身节点feature和的平均值（因为进行了一次归一化）

这就和GraphSAGE的一层Mean Aggregator没有区别，所以可以认为，一层GCN就是一层把所有邻居都考虑进去（因为可以不是全部）的并使用Mean Aggregator的GraphSAGE