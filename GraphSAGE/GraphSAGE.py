import torch
import numpy as np

# GraphSAGE model

# define the graph has 5 vertices
# random generate the adjacency list
graph = np.random.randint(0, 2, size=[5, 5])
graph = graph - np.diag(np.diag(graph))
graph = np.triu(graph)
graph += graph.T

# we define feature list is [N,1]
f = torch.from_numpy(np.random.randint(1, 11, size=[5, 1]))
f = torch.tensor(f, dtype=torch.float32)
# normalization
f = f / torch.norm(f, p=2)

# require_output
require_output = torch.tensor(torch.from_numpy(np.random.randint(0, 10, size=[5, 1])), dtype=torch.float32)
# normalization
require_output = require_output / torch.norm(require_output, p=2)

# define K=2 and select 2 neighbors to combine next self layer

# layer 1:use Mean aggregator
Z1 = torch.tensor([])
W1 = torch.tensor(torch.from_numpy(np.random.rand(1)), dtype=torch.float32)
W1.requires_grad_()

for i in range(5):
    itemSum = 0
    k = 0
    for j in range(2):
        while True:
            if k >= 5:
                k -= 5
            if k != i and graph[i][k] == 1:
                itemSum += f[k]
                k += 1
                break
            k += 1
    itemMean = (f[i] + itemSum) / 3
    t = itemMean * W1
    Z1 = torch.cat((Z1, torch.sigmoid(t)), 0)
Z1 = Z1 / torch.norm(Z1, 2)
Z1 = Z1.view(5, 1)

# layer 2:use Pooling aggregator(max)
Z2 = torch.tensor([])
W2_pool = torch.tensor(torch.from_numpy(np.random.rand(1)), dtype=torch.float32)
b = torch.tensor(torch.from_numpy(np.random.rand(1)), dtype=torch.float32)
W2_aggregate = torch.tensor(torch.from_numpy(np.random.rand(2, 1)), dtype=torch.float32)
W2_pool.requires_grad_()
W2_aggregate.requires_grad_()
b.requires_grad_()

for i in range(5):
    itemSum = 0
    itemResult = 0
    k = 0
    for j in range(2):
        while True:
            if k >= 5:
                k -= 5
            if k != i and graph[i][k] == 1:
                itemResult = Z1[k] * W2_pool + b
                if itemResult.data > itemSum:
                    itemSum = itemResult
                k += 1
                break
            k += 1
    itemResult = torch.sigmoid(itemResult)
    concat = torch.cat((Z1[i], itemResult), 0)
    concat = concat.view(1, 2)
    t = torch.mm(concat, W2_aggregate)
    Z2 = torch.cat((torch.sigmoid(t), Z2), 0)
Z2 = Z2 / torch.norm(Z2, 2)

# loss func
loss = torch.abs(require_output - Z2).sum()
loss.backward()

# gradient descent
with torch.no_grad():
    # lr = 0.1
    W1 += W1 * W1.grad
    W2_aggregate += W2_aggregate * W2_aggregate.grad
    W2_pool += W2_pool.grad
    W1.grad.data.zero_()
    W2_pool.grad.data.zero_()
    W2_aggregate.grad.data.zero_()
