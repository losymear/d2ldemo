import torch
def f(a):
    b = a * a + abs(a)
    c = b ** 3 - b ** (-4)
    return c


a = torch.randn(size=(3, 1), requires_grad=True)
print(a.shape)
print(a)
d = f(a)
d.sum().backward()
print(a.grad)
