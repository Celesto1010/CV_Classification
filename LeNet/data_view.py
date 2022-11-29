"""
看一下Cifar10数据集
"""


from torchvision import datasets, transforms
import matplotlib.pyplot as plt


transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# 下载/加载数据。下载的是经过transformer转换的数据
train_dataset = datasets.CIFAR10(
    root=r'E:\CV\Datasets\CIFAR10\train',
    train=True,
    download=False,
    transform=transformer
)
eval_dataset = datasets.CIFAR10(
    root=r'E:\CV\Datasets\CIFAR10\eval',
    train=False,
    download=False,
    transform=transformer
    )


# 将保存的数据转换为正常的图片
def convert_data_to_image(one_data):
    data_tensor, data_label = one_data[0], one_data[1]
    # 反标准化
    data_tensor = data_tensor * 0.5 + 0.5
    data_image = transforms.ToPILImage().__call__(data_tensor)
    data_label = train_dataset.classes[data_label]
    return data_image, data_label


image, label = convert_data_to_image(train_dataset[15])
image.show()
print(label)