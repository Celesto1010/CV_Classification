"""
Vgg网络——使用了多层小卷积核网络叠加，代替一层大卷积核的网络（如堆叠两个3x3卷积核代替5x5卷积核、堆叠三个3x3卷积核代替7x7卷积核）
多层小卷积核网络的感受野和一层大卷积核网络相同，同时参数量变少
感受野计算公式：F(i) = (F(i+1) - 1) × stride + Ksize. 其中F(i)为第i层的感受野，Ksize为核尺寸（在vgg中，卷积核的stride默认为1）

vgg模型共有4种参数配置。
所有卷积核stride=1、padding=1（若size=3，则不改变特征图的尺寸），所有maxpool层size=2，stride=2
"""


import torch
import torch.nn as nn


# 配置4种不同参数的vgg的结构。
# 数字代表一层卷积层，数字的值即卷积核深度，卷积核尺寸固定为3（不改变特征图大小）
# “M”代表一层最大池化层。不论哪一种参数配置，都是5个最大池化层。
# 因此特征图尺寸都是从初始的[3, 224, 224]到最终的[512, 7, 7]
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# 根据固定的参数配置，生成vgg的全连接层之前的网格
def make_layers(cfg, batch_norm: bool = False) -> nn.Sequential:
    layers = []                 # layers存储网格结构
    in_channels = 3             # 初始输入的深度固定为3
    for v in cfg:
        # M代表最大池化层
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 数字代表卷积层。卷积层后固定跟一个ReLU
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            # 若添加batch norm，则卷积层与ReLU之间添加一层nn.BatchNorm2d
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Vgg(nn.Module):
    def __init__(self, features, num_classes=1000, initialize_weights=False):
        """
        :param features: 全连接层之前的网络结构
        :param num_classes: 分类数
        """
        super().__init__()
        self.features = features
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))       # torch官方文档带一个自适应平均池化层。这里不用了
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, num_classes),
        )
        if initialize_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_vgg(model_name, **kwargs):
    cfg = cfgs.get(model_name, None)
    if not cfg:
        raise Exception(' Wrong model name. ')
    features = make_layers(cfg)
    vgg = Vgg(features, **kwargs)
    return vgg


if __name__ == '__main__':
    test_input = torch.randn(16, 3, 224, 224)
    vgg_model = make_vgg('vgg16')
    output = vgg_model(test_input)
    print(output.shape)