import torch
import torch.nn.functional as F
import numpy as np

# define the graph has 8 vertices
# random generate the adjacency list
graph = np.random.randint(0, 2, size=[8, 8])
graph = graph - np.diag(np.diag(graph))
graph = np.triu(graph)
graph += graph.T
baseGraph = torch.tensor(torch.from_numpy(graph), dtype=torch.float32)  # use for method 3

# calculate the degree
degree = list()
for i in graph:
    degree_sum = 0
    for j in i:
        degree_sum += j
    degree.append(degree_sum)
graph = np.diag(degree) - graph

# normalize the matrix
item = np.diag(graph)
# Symmetric normalized Laplacian is D^-0.5*(D-A)*D^-0.5
item = item ** -0.5

# do not select Random walk normalized Laplacian which is D^-1*(D-A)

graph = graph.astype(np.float64)
# TODO can not multiple the matrix directly.If so, will get 0.
for i in range(graph.shape[0]):
    for j in range(graph.shape[1]):
        graph[i][j] = graph[i][j] * item[j] * item[i]

# get eigVal and eig
eigVal_n, eig_n = np.linalg.eig(graph)
eigVal = torch.from_numpy(eigVal_n)
eig = torch.from_numpy(eig_n)
eigVal = torch.tensor(eigVal, dtype=torch.float32)
eig = torch.tensor(eig, dtype=torch.float32)

# suppose the feature matrix is [N,1]
f = torch.from_numpy(np.random.randint(1, 11, size=[8, 1]))
f = torch.tensor(f, dtype=torch.float32)

require_output = torch.tensor(torch.from_numpy(np.random.randint(0, 10, size=[8, 1])), dtype=torch.float32)

# **************
# Spectral Networks and Locally Connected Networks on Graphs
initList = torch.rand(8)
initCore = torch.diag(initList)
initCore.requires_grad_()

a = torch.mm(eig, initCore)
b = torch.mm(a, eig.T)
c = torch.mm(b, f)
output = F.relu(c)

loss = torch.abs(require_output - output).sum()
loss.backward()

with torch.no_grad():
    for i in range(8):
        # lr = 0.1
        initCore[i][i] += initCore.grad[i][i] * 0.1
        initCore.grad.data.zero_()

# **************

# **************
# chebynet
# the number of parameters of core K normally is far less than N , so we select 3
initCore2 = torch.rand(3)
initCore2.requires_grad_()

L = torch.tensor(torch.from_numpy(graph), dtype=torch.float32)
d = torch.mm((initCore2[0] * torch.eye(8, 8) + initCore2[1] * L + initCore2[2] * torch.pow(L, 2)), f)
output2 = torch.relu(d)

loss = torch.abs(require_output - output2).sum()
loss.backward()

with torch.no_grad():
    # lr = 0.1
    initCore2 += initCore2.grad * 0.1
    initCore2.grad.data.zero_()

# **************

# **************
# GCN
A2 = baseGraph + torch.eye(8, 8)
degree2 = list()
for i in A2:
    Sum = 0
    for j in i:
        Sum += j
    degree2.append(Sum)

for i in range(8):
    degree2[i] = degree2[i] ** -0.5

D2 = torch.diag(torch.tensor(degree2))
# feature is [N,1] and the output is [N,1], so the core is [1,1]
initCore3 = torch.rand(1, 1)
initCore3.requires_grad_()

e = torch.mm(D2, A2)
g = torch.mm(e, D2)
h = torch.mm(g, f)
output3 = F.relu(torch.mm(h, initCore3))

loss = torch.abs(require_output - output3).sum()
loss.backward()

with torch.no_grad():
    # lr = 0.1
    initCore3 += initCore3.grad * 0.1
    initCore3.grad.data.zero_()

# **************