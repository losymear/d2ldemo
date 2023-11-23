import random
import torch
from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)



#
# ## 8.3.4.1 随机采样
# def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
#     """使用随机抽样生成一个小批量子序列"""
#     # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
#     corpus = corpus[random.randint(0, num_steps - 1):]
#     # 减去1，是因为我们需要考虑标签
#     num_subseqs = (len(corpus) - 1) // num_steps
#     # 长度为num_steps的子序列的起始索引
#     initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
#     # 在随机抽样的迭代过程中，
#     # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
#     random.shuffle(initial_indices)
#     print(initial_indices)
#
#     def data(pos):
#         # 返回从pos位置开始的长度为num_steps的序列
#         return corpus[pos: pos + num_steps]
#
#     num_batches = num_subseqs // batch_size
#     for i in range(0, batch_size * num_batches, batch_size):
#         # 在这里，initial_indices包含子序列的随机起始索引
#         initial_indices_per_batch = initial_indices[i: i + batch_size]
#         # （了解代码的测试日志）
#         print(initial_indices_per_batch)
#         X = [data(j) for j in initial_indices_per_batch]
#         Y = [data(j + 1) for j in initial_indices_per_batch]
#         yield torch.tensor(X), torch.tensor(Y)
#
#
my_seq = list(range(35))
# for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)



## 8.3.4.2  顺序采样
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    # 了解代码的注释（注意运行看日志）
    # batch_size是批次数。比如batch_size=2时，Xs的shape为2*x 也就是说Xs的第一行都是一个批次。
    # 下面的for循环打印，每次都打印出所有2个批次的第1组数据、第2组数据、第3组数据.....
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    print('num_batch',num_batches,Xs.shape[1],num_steps,batch_size)
    print(Xs)
    print(Ys)

    # exit(1)
    for i in range(0, num_steps * num_batches, num_steps):
        print(i)
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)