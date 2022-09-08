import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, download=False,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


# 定义网络结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.flatten = nn.Flatten() # 好像说是1×1的卷积核做卷积
        # self.linear1 = nn.Linear(64*4*4, 64)
        # self.linear1 = nn.Linear(64, 10)
        self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Flatten(), nn.Linear(64*4*4, 64), nn.Linear(64, 10))

    def forward(self, x):
        x = self.model(x)
        return x


cnn = CNN()
loss = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = cnn(imgs)
    rl = loss(outputs, targets)
    print(rl)