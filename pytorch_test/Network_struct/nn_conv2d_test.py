import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, download=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=False)


# 定义网络
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2) # dilation 空洞卷积
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(8640, 100)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.linear1(x)
        return x


# 初始化使用网络
mynet = Mynet()

step = 0
writer = SummaryWriter("../logs_maxpool")
for data in dataloader:
    imgs, label = data
    output = mynet.forward(imgs)
    # linear_output = torch.reshape(output, (1, 1, 1, -1))
    linear_output = torch.flatten(output)
    print(linear_output.shape)
    print(output.shape)
    # 将output改为3通道数，可以在transboard中进行显示
    # output = torch.reshape(output, (-1, 3, 15, 15))
    # # print(output.shape)
    # writer.add_images("input", imgs, step)
    # writer.add_images("output", output, step)
    # step += 1

writer.close()


# 卷积层-提取特征 池化层-加快训练速度 非线性激活-引入非线性特征，提高泛化能力
# 正则化层 丢弃层-dropout防止过拟合