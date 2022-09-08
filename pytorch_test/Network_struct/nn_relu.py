import torch
from torch import nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)


class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.relu1 = nn.ReLU()
        # Sigmoid()

    def forword(self, x):
        x = self.relu1(x)
        return x

