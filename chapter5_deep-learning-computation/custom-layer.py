# 自定义层 https://zh-v2.d2l.ai/chapter_deep-learning-computation/custom-layer.html

import torch
import torch.nn.functional as F
from torch import nn


## 5.4.1 不带参数的层


### CenteredLayer的作用是将输入减去它的均值
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


### 使用示例
layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

### 可以与其它模块进行组合
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()



## 5.4.2 带参数的层

### 自定义全连接层，in_units 输入数 units 输出数。  使用ReLU作为激活函数
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight)
