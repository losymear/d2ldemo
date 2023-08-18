# 3.3 线性回归的简洁实现

import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


## 3.3.2数据集读取

def load_array(data_arrays, batch_size, is_train=True):  # @save
    """
    :param is_train:  布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
    构造一个PyTorch数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# next(iter(data_iter))


### 3.3.3 定义模型
# 对于标准深度学习模型，我们可以使用框架的预定义好的层。这使我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。
# 我们首先定义一个模型变量net，它是一个Sequential类的实例。
# Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。
# 在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”。

# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
