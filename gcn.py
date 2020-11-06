import torch
import torch.nn.functional as F
import numpy as np

# define the graph has 8 vertices
# random generate the adjacency list
graph = np.random.randint(0, 2, size=[8, 8])
graph = graph - np.diag(np.diag(graph))
graph = np.triu(graph)
graph += graph.T

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
item = item ** -1.0
graph = graph.astype(np.float64)
# TODO can not multiple the matrix directly.
for i in range(graph.shape[0]):
    for j in range(graph.shape[1]):
        graph[i][j] = graph[i][j] * item[j]

# get eigVal and eig
eigVal, eig = np.linalg.eig(graph)

# suppose the feature matrix is [N,3]
f = np.random.randint(1, 11, size=[8, 3])

# f' = U^T^ * f
f_i = eig.T * f

# consider a 10 classification problem
out = np.random.randint(0, 10, [8, 1])

# tensor_input,tensor_output
t_input = torch.from_numpy(f_i)
t_input = t_input.permute(1, 0)  # batch * N
t_output = torch.from_numpy(out)


# deal with the convolution operation
class ConvolutionNet(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # 2 convolution layer
        self.conv1 = torch.nn.Conv1d(3, 4, 3)
        self.conv2 = torch.nn.Conv1d(4, 8, 2)

        # 2 fully connected layer
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 10)

    def forward(self, x):
        r = self.conv1(x)
        r = F.relu(r)
        r = F.max_pool1d(r, 2)
        r = self.conv2(r)
        r = F.relu(r)
        r = F.max_pool1d(r, 2)
        r = r.view(8)
        r = F.relu(self.fc1(r))
        r = self.fc2(r)
        return r


cNet = ConvolutionNet()
opt = torch.optim.SGD(cNet.parameters(), lr=0.1)

# recurrence part
opt.zero_grad()
out = cNet(t_input)
cFun = torch.nn.MSELoss()
loss = cFun(out, t_output)
loss.backward()
opt.step()
