import torch
import numpy as np
def gather_by_index(src, idx, dim=1, squeeze=True):
    """Gather elements from src by index idx along specified dim

    Example:
    >>> src: shape [64, 20, 2]
    >>> idx: shape [64, 3)] # 3 is the number of idxs on dim 1
    >>> Returns: [64, 3, 2]  # get the 3 elements from src at idx
    """
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    return src.gather(dim, idx).squeeze() if squeeze else src.gather(dim, idx)

src = torch.full([64,20,2],3)
idx = torch.full([64,3],2)

#print(gather_by_index(src,idx))
#print(src.size(2))
x = [[1,2,3],[4,5,6]]
y = [[7,8,9],[0,1,2]]

avg=np.array(x)+np.array(y)
avg=(avg/2).tolist()
print(avg)
avg=(-np.array(avg)).tolist()
print(avg)

a = 10
for i in range(5):
    i = i-1
print(a)

Q = [[1,1,1],[2,2,2]]
V = [[np.exp(3),np.exp(3),np.exp(3)],[np.exp(4),np.exp(4),np.exp(4)]]
Q = torch.tensor(Q)
V = torch.tensor(V)
l = []
for q,v in zip(Q,V):
    loss = -q*torch.log(v)
    print(loss)
    l.append(loss)

b = sum(l)
print(b)
a = torch.tensor([1,2,3])
print(a/3)

x = torch.tensor([2, 3, 4], dtype=torch.float, requires_grad=True)
print(x)
y = x * 2
while y.norm() < 1000:
    y = y * 2
print(y)

y.backward(torch.ones_like(y))
print(x.grad)
