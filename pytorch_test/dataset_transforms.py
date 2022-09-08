# 给索引，告诉程序数据集在什么位置
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.utils.tensorboard import SummaryWriter

dataset_compose = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_images = torchvision.datasets.CIFAR10(train=True, root="../data", transform=dataset_compose, download=True)
test_images = torchvision.datasets.CIFAR10(train=False, root="../data", transform=dataset_compose, download=True)

# d2l.show_images([test_images[i][0] for i in range(32)], 4, 8, scale=0.8)
# d2l.plt.show()
# print(test_images[0])
# 我们在使用时要转化为tensor的数据类型，那该怎么转呢？

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_images[i]
    writer.add_image("test_images", img, i)
    print(img.shape)
