"""
分离train、test文件的方法
适用于原始数据形式为 ./root/class_a/...、 ./root/class_b/...的形式
分离为适用于torchvision.datasets.ImageFolder的形式
"""

import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main(root_dir, target_dir, split_rate=0.1):
    # 保证随机可复现
    random.seed(0)
    assert os.path.exists(root_dir), "path '{}' does not exist.".format(root_dir)

    classes = [cla for cla in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, cla))]

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    # 建立保存训练集的文件夹
    train_dir = os.path.join(target_dir, "train")
    mk_file(train_dir)
    # 每个类别都建立文件夹
    for cla in classes:
        mk_file(os.path.join(train_dir, cla))

    # 建立保存验证集的文件夹
    val_dir = os.path.join(target_dir, "val")
    mk_file(val_dir)
    for cla in classes:
        mk_file(os.path.join(val_dir, cla))

    # 对原始数据的每个类别的图像分别进行随机采样
    for cla in classes:
        cla_path = os.path.join(root_dir, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(val_dir, cla)
                copy(image_path, new_path)
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_dir, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    main(root_dir=r'E:\PythonProjects\CV\Datasets\flower_photos',
         target_dir=r'E:\PythonProjects\CV\Datasets\flower_photos_for_train',
         split_rate=0.2)
