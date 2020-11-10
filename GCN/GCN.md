## GCN

#### Spectral Method

卷积核为对角矩阵，因为对于[N,1]的feature来说，U^T^f与U^T^ c应该能够内积，而这与对角矩阵在数学运算上等同。

为什么使用renormalization而不是random walk normalization，是因为需要保证L为对称矩阵，而具有N个正交特征向量。

#### Chebynet

使用切比雪夫多项式来逼近，不用计算特征向量了。

$G(\Lambda)$的表达式可以写为

$$G(\Lambda) \approx \sum_{k=0}^K \theta_kT_K(\widetilde\Lambda)$$ 

其中$\widetilde\Lambda = 2\Lambda/\lambda_{max}-I_N$，式子最后可以写成
$$
OutPut = \sum_{k=0}^K\theta_kT_K(\widetilde L)x
$$
这里的$\widetilde L$也是经过了类似的变换，但是可以理解为
$$
f*_Gg = \sum_{k=0}^{K-1}\theta_kL^kx
$$
因为，$T_K$作为切比雪夫多项式是可以认为是一组线性无关的基底的。

#### GCN

就是取K=1的Chebynet，并且增加了$\theta_0 =\theta_1$，这是为了防止参数过拟合。

最后的公式是
$$
Z = \widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}X\Theta
$$
其中
$$
I_N + D^{-1/2}AD^{-1/2}\stackrel{\widetilde A = A + I_N}\longrightarrow \widetilde D^{-1/2}\widetilde A\widetilde D^{-1/2}
$$

$$
\widetilde A = A+I_N,\widetilde D_{ii}=\sum_j\widetilde A_{ij}
$$

