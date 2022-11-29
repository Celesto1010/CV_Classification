"""
观察flower_photos数据集。
"""

from torchvision import transforms
from PIL import Image
import os
import random


transformer = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'eval': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
}

data_dir = r'E:\PythonProjects\CV\Datasets\flower_photos'
categories = os.listdir(data_dir)

# 随机取一张，并将其以train的方式转为tensor
category = random.choice(categories)
category_dir = os.path.join(data_dir, category)
pictures = os.listdir(category_dir)
picture = random.choice(pictures)
picture_path = os.path.join(category_dir, picture)

image = Image.open(picture_path)
image.show()
image_tensor = transformer['train'](image)
print(category)
print(image_tensor.shape)

# 将tensor变为图像（仅执行de-normalization），并显示
tensor = image_tensor * 0.5 + 0.5
transformed_image = transforms.ToPILImage().__call__(tensor)
transformed_image.show()


