# 网络模型
# 损失loss
# 数据 (这个需要重新赋值)
# 1.这些后面加.cuda()就能运行
# 2.device = torch.device("cpu/cuda:0") 然后后面加.to(device)就行
# image.convert('RGB')
# 在gpu上训练的模型，需要放到CPU上运行，就要改一下代码map_location=torch.device('cpu')
# net.train() net.eval()

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备数据集

train_data = torchvision.datasets.CIFAR10(root="../../data", train=True, download=False,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="../../data", train=False, download=False,
                                          transform=torchvision.transforms.ToTensor())

# 加载数据集
train_data_size = len(train_data)
test_data_size = len(test_data)
train_data_load = DataLoader(train_data, batch_size=64)
test_data_load = DataLoader(test_data, batch_size=64)

# 设置训练位置
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# 定义网络结构
class WangLr(nn.Module):
    def __init__(self):
        super(WangLr, self).__init__()
        self.module = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(),
                                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Flatten(), nn.Linear(64*4*4, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        x = self.module(x)
        return x


net = WangLr().to(device)
# 损失函数
loss = nn.CrossEntropyLoss().to(device)
# 优化器
optim = torch.optim.SGD(net.parameters(), lr=0.01)
# 设置网络的一些参数
train_epoch = 40
train_times = 0
# 开始训练
writer = SummaryWriter("../train_nn")
print("let's begin----")
for i in range(train_epoch):
    nums = 0
    total_train_loss = 0
    for data in train_data_load:
        train_imgs, train_labels = data  # 载入数据
        train_imgs = train_imgs.to(device)
        train_labels = train_labels.to(device)
        outputs = net(train_imgs)   # 输入网络进行训练
        train_loss = loss(outputs, train_labels)    # 计算损失
        optim.zero_grad()   # 在求节点梯度之前，需要将之前保留的梯度设为0
        train_loss.backward()    # 在优化参数之前，需要反向传播求出节点梯度
        optim.step()    # 开始调优
        nums += 1
        if nums % 100 == 0:
            print(train_loss.item())

        total_train_loss += train_loss.item()

    writer.add_scalar("train_loss", total_train_loss/5, i+1)
    print("训练第{}轮，loss为：{}".format(i+1, total_train_loss/5))

    # 每一轮进行一次验证，测试数据集的loss以及accuracy，注意这个时候只做验证，不改变参数，因此梯度设置不变
    total_accuracy = 0
    total_test_loss = 0
    with torch.no_grad():
        for test_data in test_data_load:
            test_imgs, test_labels = test_data
            test_imgs = test_imgs.to(device)
            test_labels = test_labels.to(device)
            outputs = net(test_imgs)
            test_loss = loss(outputs, test_labels)
            accuracy = (outputs.argmax(1) == test_labels).sum() # argmax函数的作用就是挑出行中的最大值，以其下标为值
            total_test_loss += test_loss.item()
            total_accuracy += accuracy

        writer.add_scalar("test_loss", total_test_loss, i + 1)
        writer.add_scalar("accuracy", total_accuracy/test_data_size, i + 1)
        print("测试第{}轮，loss为：{}".format(i+1, total_test_loss))
        print("测试第{}轮，accuracy为：{}".format(i + 1, total_accuracy/test_data_size))

        if i % 10 == 0:
            print("训练第{}轮模型已存入".format(i))
            torch.save(net, "./models/WangLr_{}.pth".format(i))

writer.close()