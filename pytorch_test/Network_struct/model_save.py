import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1 模型结构，参数都有保存
torch.save(vgg16, "./models/vgg16_m1.pth")

# 保存方式2 保存模型的参数（官方推荐的保存方式）
torch.save(vgg16.state_dict(), "./models/vgg16_m2.pth")


# 陷阱
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
torch.save(cnn, "./models/cnn_m1.pth")
