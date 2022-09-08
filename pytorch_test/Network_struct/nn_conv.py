import torch
import torch.nn.functional as F

ori =   [[1, 2, 0, 3, 1],
        [0, 1, 2, 3, 1],
        [1, 2, 1, 0, 0],
        [5, 2, 3, 1, 1],
        [2, 1, 0, 1, 1]]

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [0, 1, 0]])

# 不满足卷积操作时的尺寸，此处引入尺寸变换吧
input = torch.reshape(input, (1, 1, 5, 5)) # batchsize channel size
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(input.shape)
print(input)
# 注意F.是卷积操作 nn.是卷积层
output = F.conv2d(input, kernel, stride=1)
print(output)

print(ori)