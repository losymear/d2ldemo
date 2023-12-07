import torch
from d2l import torch as d2l


# 平均汇聚层可以被视为输入的加权平均值， 其中各输入的权重是一样的。
# 实际上，注意力汇聚得到的是加权平均的总和值， 其中权重是在给定的查询和不同的键之间计算得出的。


# @save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            print(matrix.shape)
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)


attention_weights = torch.eye(10).reshape((1, 1, 10, 10))
show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')




# https://zh-v2.d2l.ai/chapter_attention-mechanisms/attention-cues.html 练习2 生成随机概率并画热力图。
# from torch.nn import functional as F
# X = torch.rand((10,10))
# X = F.softmax(X,dim=1)
# X = X.reshape((1,1,10,10))
# show_heatmaps(X,xlabel='Columns',ylabel='Rows')