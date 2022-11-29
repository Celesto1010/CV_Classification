"""
LeNet网络结构：
两层卷积+池化层，后添加三层全连接层
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 第一组卷积 - 池化层。深度调整为16，
        # 卷积层采用kernel_size=5, padding=0，stride=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))
        self.pool1 = nn.MaxPool2d(2, 2)
        # 第二组卷积 - 池化层。深度调整为32
        # 卷积层采用kernel_size=5, padding=0, stride=1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.pool2 = nn.MaxPool2d(2, 2)
        # 三层全连接层. 经过前面一层后，输入的维度变为32, 5, 5
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # 每次经过卷积层后，都再经过一层relu激活函数
        # [3, 32, 32] → [16, 28, 28]
        x = F.relu(self.conv1(x))
        # [16, 28, 28] → [16, 14, 14]
        x = self.pool1(x)
        # [16, 14, 14] → [32, 10, 10]
        x = F.relu(self.conv2(x))
        # [32, 10, 10] → [32, 5, 5]
        x = self.pool2(x)
        # 从第一维开始展平
        x = torch.flatten(x, 1)
        # 每经过一个全连接层，也过一层relu
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 经过最后一层全连接层后，不需要再relu
        return self.fc3(x)


if __name__ == '__main__':
    test_input = torch.randn(size=[10, 3, 32, 32])
    model = LeNet(10)
    output = model.forward(test_input)
    print(output.shape)         # 10x10
