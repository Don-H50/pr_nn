import torch
import torchvision

# 保存方式1 的模型加载方式
from torch import nn

vgg16 = torch.load("./models/vgg16_m1.pth")

# 保存方式2 的模型加载方式
# vgg16_parameter = torch.load("./models/vgg16_m1.pth") 输出的是字典形式
vgg16_1 = torchvision.models.vgg16(pretrained=False)
vgg16_1.load_state_dict(torch.load("./models/vgg16_m2.pth"))


print(vgg16)
print(vgg16_1)

# 陷阱1(还是会需要将模型import进来)
from model_save import CNN
cnn = torch.load("./models/cnn_m1.pth")
print(cnn)