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
optim = torch.optim.SGD(cnn.parameters(), lr=0.01)  # 我把网络中的参数放到优化器中，这样可以计算梯度
for data in dataloader:
    imgs, targets = data
    outputs = cnn(imgs)
    rl = loss(outputs, targets) # 求出损失
    # 开始优化
    optim.zero_grad()   # 梯度调为0
    rl.backward()   # 利用反向传播求出每个节点的梯度
    optim.step()    # 利用优化器就对每个模型参数进行调优

