import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

vgg16_false = torchvision.models.vgg16(pretrained=False) # 仅加载网络模型
# vgg16_true = torchvision.models.vgg16(pretrained=True) # 参数都得下载下来

# print(vgg16)发现该模型输出的是1000种分类，那如果我加载的是CIFAR10数据集，只有10种分类，该如何使用这个网络呢？
dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, download=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=False)


# vgg16_false.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
for data in dataloader:
    imgs, labels = data
    print(labels)
    print(labels.shape)