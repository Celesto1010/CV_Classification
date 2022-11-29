"""
AlexNet:
首次使用GPU进行训练；
使用了ReLU激活函数，而不是传统的Sigmoid或Tanh
使用了LRN局部相应归一化
全连接层的前两层中使用了dropout，防止过拟合

模型结构：
两层 卷积 - ReLU - 池化
三层 卷积 - ReLU
一层 池化
一层 平均池化(torch官方代码中有，此代码调整了参数，故去掉了这一层)
三层 全连接层
"""


import torch
import torch.nn as nn


# 这里写的与torch官方的alexnet的参数有少许不同，但是整体结构相同
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # input维度默认为[3, 224, 224]
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, padding=2, stride=4),      # [3, 224, 224] → [48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # [48, 55, 55] → [48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),               # [48, 27, 27] → [128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # [128, 27, 27] → [128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),              # [128, 13, 13] → [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),              # [192, 13, 13] → [192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),              # [192, 13, 13] → [128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)                       # [128, 13, 13] → [128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    test_input = torch.randn(10, 3, 224, 224)
    alexnet = AlexNet()
    output = alexnet(test_input)
    print(output.shape)