import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# 其他激活函数
# def SquarePlus(x, b=4 * (np.log(2)) ** 2):
#     res=0.5 * (x + (x ** 2 + b).sqrt())
#     return res

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # x = x.to(torch.float32)
        y = x.view(x.shape[0], -1)
        print(x.shape, y.shape)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        return y


class Logisticmodel(nn.Module):
    def __init__(self):
        super(Logisticmodel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # x = x.to(torch.float32)
        y = x.view(x.shape[0], -1)
        print(x.shape, y.shape)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        return y


# 定义网络结构
# 定义AlexNet网络结构
class AlexNet(nn.Module):
    def __init__(self, width_mult=1):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(  # 输入1*28*28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32*28*28
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32*14*14
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64*14*14
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*7*7
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128*7*7
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256*7*7
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 256*7*7
            nn.MaxPool2d(kernel_size=3, stride=2),  # 256*3*3
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# class Model_squre(nn.Module):
#     def __init__(self):
#         super(Model_squre, self).__init__()
#         self.fc1 = nn.Linear(784, 256)
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 10)
#         self.relu4 = nn.ReLU()
#
#     def forward(self, x):
#         # x = x.to(torch.float32)
#         y = x.view(x.shape[0], -1)
#         print(x.shape, y.shape)
#         y = self.fc1(y)
#         y = Relu(y)
#         y = self.fc2(y)
#         y = Relu(y)
#         return y






