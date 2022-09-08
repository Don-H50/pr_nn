from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
# writer.add_image("train_img", img_array, nums, dataformats = 'HWC') 这个意思是没有转换为tensor形式就要标记为HWC
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)


writer.close()