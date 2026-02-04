import torch

#prob = torch.tensor([[0.8, 0.1, 0.1, 0.0],[0.25, 0.25, 0.25, 0.25],[0,0,0,1]])
prob = torch.tensor([[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25]
,[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25],[0.25, 0.25, 0.25, 0.25]])
selected = torch.multinomial(prob, num_samples=1)
selected2 = torch.multinomial(prob, num_samples=1).squeeze(1)
print(selected)
print(selected2)
                # shape: (batch,)

### 
'''对于torch.multinomial()来说，即使每个batch的概率分布一样，每个batch也会随机选择不同的动作

test的时候只关注最后的makespan，不关注中间的结果，
所以不同长度的trajectory可以并在一起计算

train的时候需要对每个t时刻进行计算，长度不同的trajectory如果并在一起，
已经结束的trajectory也会被计算，

'''