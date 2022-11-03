import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import torchvision
import matplotlib.pyplot as plt
import torchsummary as summary

import nn as cnn


prob = 0.2
dropout = nn.Dropout(p=prob)
x = torch.ones(10)
print(x)
y = dropout(x)
print(y)


def fx(x):
    return x**2 - 6*x + 1


def deriv(x):
    return 2*x - 6


x = np.linspace(-14, 20, 2000)
localmin = np.random.choice(x, 1)

epoch = 1000
lr = 0.01

# for i in range(epoch):
#     grad = deriv(localmin)
#     delta = lr * grad
#     localmin -= delta

# plt.plot(x, fx(x))
# plt.plot(localmin, fx(localmin), 'ro')
# plt.show()

net = cnn.Net()
tensor = torch.randn(1, 1, 28, 28)
net(tensor)
print(net)
for p in net.named_parameters():
    # print(p)
    pass

n_params = 0
for p in net.parameters():
    if p.requires_grad:
        n_params += p.numel()

print(f'{n_params=}')

summary.summary(net, input_size=(1, 28, 28))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize(size=(120, 80)),
    torchvision.transforms.ToTensor(),
    ])


np_img = cv.imread('data/train_images/free/free_470.png')

tensor = transform(np_img)
tensor = tensor.unsqueeze(0)
conv = nn.Conv2d(3, 4, 3, stride=1, padding=1)
out = conv(tensor)
print(f'{out.shape=}')
print(f'{np_img.shape=}')
print(f'{tensor.shape=}')

# print(f'{np_img.shape=}')
# tensor = torch.Tensor(np_img)
# tensor = tensor.permute(2, 0, 1)
# tensor = tensor.unsqueeze(0)
# print(f'{tensor.shape=}')
# conv = nn.Conv2d(3, 4, 3, stride=1)
# out = conv(tensor)
# print(f'{out.shape=}')
