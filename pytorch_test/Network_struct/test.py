import torch
import torchvision
from PIL import Image
from torch import nn
from models.nn_model import WangL, WangLr
# con = nn.Conv1d()

net = torch.load("./models/WangL_31.pth")
net_1 = torch.load("./models/WangLr_50.pth")
image_path = "../../data/model_test/WangL/bird1.jpg"

img = Image.open(image_path)
print(img.size)
print(img)
# image = image.convert('RGB')
img_totensor = torchvision.transforms.ToTensor()
img_resize = torchvision.transforms.Resize((32, 32))
compose = torchvision.transforms.Compose([img_totensor, img_resize])
img = compose(img) # 得到 3， 32， 32 tensor形式的img
print("look at me", img.shape)
print(img)
img = torch.reshape(img, (1, 3, 32, 32)).cuda()

output = net(img)
output_1 = net_1(img)

print(output, '\n', output_1)
print(output_1.argmax(1).item())




#
# img = Image.open(image_path)
# img.show()
# totensor = torchvision.transforms.ToTensor()
# resize = torchvision.transforms.Resize((32, 32))
# img = resize(totensor(img))
# # img = torch.reshape(img, (1, 3, 32, 32))
# # img.show()
# print(img.shape)
# topil = torchvision.transforms.ToPILImage()
# PIL_img = topil(img)
# PIL_img.show()
