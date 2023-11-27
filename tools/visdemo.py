# import torch
from torch import nn
# from torchviz import make_dot
# import torchvision.models as models
#
# resnet = models.resnet50()
# x = torch.zeros(1, 3, 224, 224, dtype=torch.float, requires_grad=False)
# out = make_dot(resnet(x)).render("rnn_torchviz", format="png", view="True")
#

import torchvision.models as models

from torchview import draw_graph

model_graph = draw_graph(models.resnet50(), input_size=(1, 3, 224, 224), expand_nested=True)
# model_graph = draw_graph(nn.RNN(100, 100), input_size=(100,100), expand_nested=True)
model_graph.visual_graph.view("rnn.png")
