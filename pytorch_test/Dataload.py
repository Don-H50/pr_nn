# 把数据加载到神经网络中，从dataset中取。如何取，取多少？设置参数进行操作就行
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_compose = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
test_images = torchvision.datasets.CIFAR10(train=False, root="../data", transform=dataset_compose, download=True)

test_loader = DataLoader(dataset=test_images, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试集中第一张图片及其target
# img, target = test_images[0]
# print(img.shape)
# print(target)

# 此时给到Loader里面的dataset要是tensoer的形式吧~
writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, label = data
    writer.add_images("loader", imgs, step)
    step = step+1

writer.close()
