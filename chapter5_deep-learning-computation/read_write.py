# @see https://zh-v2.d2l.ai/chapter_deep-learning-computation/read-write.html
# 到目前为止，我们讨论了如何处理数据， 以及如何构建、训练和测试深度学习模型。
# 然而，有时我们希望保存训练的模型， 以备将来在各种环境中使用（比如在部署中进行预测）。
# 此外，当运行一个耗时较长的训练过程时， 最佳的做法是定期保存中间结果， 以确保在服务器电源被不小心断掉时，
# 我们不会损失几天的计算结果。 因此，现在是时候学习如何加载和存储权重向量和整个模型了。


import torch
from torch import nn
from torch.nn import functional as F


## case1  张量存取
x = torch.arange(4)
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')

## case2  模型参数存取
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

### 保存
torch.save(net.state_dict(), 'mlp.params')

### 读取
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()