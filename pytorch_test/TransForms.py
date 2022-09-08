# torchtxt 可以用来转文本？
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# tensor数据类型？
# 通过transforms.ToTensor去看两个问题
# 1. transforms如何使用
# 2. tensor的数据类型有什么作用

img_path = "../data/train/ants_image/5650366_e22b7e1065.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")

print(img.size)

# 1.如何使用transforms
# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# cv.imread()
writer.add_image("train", tensor_img, 0)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
norm_img = trans_norm(tensor_img)
writer.add_image("Normalize", norm_img, 1)

# Resize 输入格式为PIL的img格式
trans_resize = transforms.Resize((512, 512))
resize_img = trans_resize(img)
resize_tensor_img = tensor_trans(resize_img)
writer.add_image("Resize", resize_tensor_img, 2)

# Compose - resize - 2 compose中的参数需要的是一个列表，数据需要transforms类型
trans_resize_2 = transforms.Resize(300)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
resize_tensor_2_img = trans_compose(img)
writer.add_image("Resize_2", resize_tensor_2_img, 0)

# RandomCrop 进行随机裁剪

writer.close()
