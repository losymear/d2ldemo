# 动手学深度学习 https://zh-v2.d2l.ai/chapter_recurrent-modern/machine-translation-and-dataset.html#subsec-mt-data-loading
# 9.5节主要是对数据进行处理，用于后续训练。

import os
import torch
from d2l import torch as d2l

# ----
# step1 9.5.1 下载数据集
# ----

# @save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


# @save
def read_data_nmt():
    """载入“英语－法语”数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()


raw_text = read_data_nmt()
print(raw_text[:75])


# ----
# step1 下载数据集
# ----


# @save
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""

    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


text = preprocess_nmt(raw_text)
print(text[:80])


# ----
# step1 9.5.2 词元化
# ----

# @save
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


source, target = tokenize_nmt(text)

#
# #@save
# def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
#     """绘制列表长度对的直方图"""
#     d2l.set_figsize()
#     _, _, patches = d2l.plt.hist(
#         [[len(l) for l in xlist], [len(l) for l in ylist]])
#     d2l.plt.xlabel(xlabel)
#     d2l.plt.ylabel(ylabel)
#     for patch in patches[1].patches:
#         patch.set_hatch('/')
#     d2l.plt.legend(legend)
#
# show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
#                         'count', source, target)


# ----
# step3 9.5.3 词表
# 由于机器翻译数据集由语言对组成， 因此我们可以分别为源语言和目标语言构建两个词表。
# 使用单词级词元化时，词表大小将明显大于使用字符级词元化时的词表大小。
# 为了缓解这一问题，这里我们将出现次数少于2次的低频率词元 视为相同的未知（“<unk>”）词元。
# 除此之外，我们还指定了额外的特定词元， 例如在小批量时用于将序列填充到相同长度的填充词元（“<pad>”）， 以及序列的开始词元（“<bos>”）和结束词元（“<eos>”）。
# 这些特殊词元在自然语言处理任务中比较常用。
# ----

src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)


# ----
# step4 9.5.4 加载数据集
# 对数据进行补全或者截断，使数据只有一个
# ----

# @save
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


# @save
# 将文本序列 转换成小批量数据集用于训练
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # 记录了每个文本序列的长度， 统计长度时排除了填充词元， 在稍后将要介绍的一些模型会需要这个长度信息。
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len

# ----
# 9.5.5 训练
# ----


#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    print(source)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
# 读取第一个小批量的数据
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break