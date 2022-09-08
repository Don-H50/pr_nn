from torch import nn


class WangL(nn.Module):
    def __init__(self):
        super(WangL, self).__init__()
        self.module = nn.Sequential(nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
                                    nn.MaxPool2d(kernel_size=2, stride=2),
                                    nn.Flatten(), nn.Linear(64*4*4, 64), nn.Linear(64, 10))

    def forward(self, x):
        x = self.module(x)
        return x


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
