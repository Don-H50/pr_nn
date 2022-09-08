import os
import cv2
import torchvision
from PIL import Image
from torch.utils.data import Dataset

# dataset = torchvision.datasets.imagenet()

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

# 可以去datasets后面看，自动从官网获取数据集，里面的官网同样是返回img and label，所以说这就是一个标准的数据集格式
    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "../data/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_datasets = MyData(root_dir, ants_label_dir)
ant_img, label = ants_datasets[2]
ant_img.show()