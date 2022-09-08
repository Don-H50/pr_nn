import torch
import torchvision
from torch import nn


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
print(cnn)
# 检验网络的正确性
input = torch.ones((64, 3, 32, 32))
output = cnn(input)
print(output.shape)