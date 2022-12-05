"""
GoogLeNet
引入了inception结构，融合了不同尺度的特征信息
使用1×1的卷积核进行降维与映射处理
添加了两个辅助分类器帮助训练
不使用全连接层，使用平均池化层，极大减少了模型参数

Inception结构：
一个1×1卷积层、一个3×3卷积层、一个5×5卷积层、一个3×3最大池化层的并联结构。输入特征图经过该并联结构，并将每一层的输出进行拼接，得到最终的输出。
将一个1×1的卷积层串联在3×3、5×5卷积层之前、3×3池化层之后，用于降维。（经过1×1卷积核可以缩小特征图深度）
最终拼接是沿着深度进行拼接，故并联的每一个分支输出的特征矩阵的尺寸需相同

辅助分类器：
5×5平均池化(stride=3) → 1×1卷积(128) with ReLU → fc(1024) with ReLU → Dropout(0.7) → fc → softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            # 两个辅助分类器分别是inception4a之后与inception4d之后
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 第一组卷积 - 池化层
        x = self.conv1(x)           # [3, 224, 224] → [64, 112, 112]
        x = self.pool1(x)           # [64, 112, 112] → [64, 56, 56]

        # 第二组卷积(降维) - 卷积 - 池化层
        x = self.conv2(x)           # [64, 56, 56] → [64, 56, 56]
        x = self.conv3(x)           # [64, 56, 56] → [192, 56, 56]
        x = self.pool2(x)           # [192, 56, 56] → [192, 28, 28]

        # inception3 - 池化层
        x = self.inception3a(x)         # [192, 28, 28] → [256, 28, 28]
        x = self.inception3b(x)         # [256, 28, 28] → [480, 28, 28]
        x = self.pool3(x)               # [480, 28, 28] → [480, 14, 14]

        # inception4 - 池化层
        # [480, 14, 14] → [512, 14, 14] → [512, 14, 14] → [512, 14, 14] → [528, 14, 14] → [832, 14, 14] → [832, 7, 7]
        x = self.inception4a(x)                 # [480, 14, 14] → [512, 14, 14]
        # inception4a后接第一个辅助分类器
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)                 # [512, 14, 14] → [512, 14, 14]
        x = self.inception4c(x)                 # [512, 14, 14] → [512, 14, 14]
        x = self.inception4d(x)                 # [512, 14, 14] → [528, 14, 14]
        # inception4d后接第二个辅助分类器
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)                 # [528, 14, 14] → [832, 14, 14]
        x = self.pool4(x)                       # [832, 14, 14] → [832, 7, 7]

        # inception5 - avgpool - linear
        x = self.inception5a(x)                 # [832, 7, 7] → [832, 7, 7]
        x = self.inception5b(x)                 # [832, 7, 7] → [1024, 7, 7]
        x = self.avgpool(x)                     # [1024, 7, 7] → [1024, 1, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.training and self.aux_logits:
            return x, (aux2, aux1)
        return x


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
        Inception模块结构.
        输出特征图的深度为ch1x1 + ch3x3 + ch5x5 + pool_proj。特征图的大小与输入特征图的大小相同
        :param in_channels: 输入特征图的深度
        :param ch1x1: 1×1卷积层深度
        :param ch3x3red: 3×3卷积层之前的，用于降维的1×1卷积层的深度
        :param ch3x3: 3×3卷积层深度
        :param ch5x5red: 5×5卷积层之前的，用于降维的1×1卷积层的深度
        :param ch5x5: 5×5卷积层的深度
        :param pool_proj: pooling层之后的，用于降维的1×1卷积层的深度
        """
        super().__init__()
        # 四个并联分支。每个分支输出的特征图大小都与输入特征图大小相同
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 沿着深度方向做拼接。最终的深度为ch1x1 + ch3x3 + ch5x5 + pool_proj
        output = [branch1, branch2, branch3, branch4]
        return torch.cat(output, 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        """ Inception后的辅助分类器 """
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # 两个inception层，其输出的维度分别为[512, 14, 14]、[528, 14, 14]
        # [d, 14, 14] → [d, 4, 4]
        x = self.avg_pool(x)
        # [d, 4, 4] → [128, 4, 4]
        x = self.conv(x)
        # [128, 4, 4] → 2048
        x = torch.flatten(x, 1)

        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.fc2(x)
        return x


class BasicConv2d(nn.Module):
    """ 卷积 + ReLU """
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    test_input = torch.randn(32, 3, 224, 224)
    googlenet = GoogLeNet()
    output, _, _ = googlenet(test_input)
    print(output.shape)
