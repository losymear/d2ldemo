# https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/sequence.html#id7

import torch
from torch import nn
from d2l import torch as d2l

## step1
# 使用正弦函数和一些可加性噪声来生成序列数据
## 时间步1,2,3,4,...1000
T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))

## step2
## 将数据转换成特征－标签（feature-label）对
# 将数据映射成Yt=Xt和Xt=[Xt-4,Xt-3,Xt-2,Xt-1]
# 只使用前600个数据

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    # 简单解释下代码（不然下次看会比较慢） features第0列为x的[0-997]，  第1列为x的[1-998].. 那么features的第一行为x的[0,1,2,3]
    features[:, i] = x[i: T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# 只有前n_train个样本用于训练
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)


### step3  使用一个简单的架构模型
# 初始化网络权重的函数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


# 一个简单的多层感知机
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net


# 平方损失。注意：MSELoss计算平方误差时不带系数1/2
loss = nn.MSELoss(reduction='none')


## step4 训练

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')


net = get_net()
train(net, train_iter, loss, 5, 0.01)

## step5 单步预测
onestep_preds = net(features)
# d2l.plot([time, time[tau:]],
#          [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
#          'x', legend=['data', '1-step preds'], xlim=[1, 1000],
#          figsize=(6, 3))
# d2l.plt.show()


## step6 多步预测（4步）
# 解释： 这里将multistep_preds的数据只保留了前n_train（即600），而600-1000之后的数据都是根据先前的预测数据生成的.
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
d2l.plt.show()


# ## step7 多步预测对比（1，4，16）
# max_steps = 64
#
# features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# # 列i（i<tau）是来自x的观测，其时间步从（i）到（i+T-tau-max_steps+1）
# for i in range(tau):
#     features[:, i] = x[i: i + T - tau - max_steps + 1]
#
# # 列i（i>=tau）是来自（i-tau+1）步的预测，其时间步从（i）到（i+T-tau-max_steps+1）
# for i in range(tau, tau + max_steps):
        # （解释。 遍历4-68。第4列是由0-3列预测出来，第5列是由1-4列预测出来....
#     features[:, i] = net(features[:, i - tau:i]).reshape(-1)
#
# steps = (1, 4, 16, 64)
# d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
#          [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
#          legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
#          figsize=(6, 3))
# d2l.plt.show()
