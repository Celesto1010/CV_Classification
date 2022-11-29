import torch
import torch.nn as nn
from model_structure import LeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs):
    """
    模型的训练（以及验证）方法
    :param model: 实例化的模型对象
    :param dataloaders: dict，形如{'train': train_loader, 'eval': eval_loader}
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param num_epochs: epoch数
    :return:
    """
    val_acc_history = []

    # 记录最优模型的参数与其对应的准确率
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 每一个epoch，分别执行一次train与eval
        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_corrects = .0, 0
            # 训练/验证过程
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs.to(device)
                labels.to(device)
                optimizer.zero_grad()

                # 仅当为训练模式时，模型更新梯度
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss
                running_corrects += torch.sum(preds == labels)

        # 每个epoch训练/验证完毕后，统计该epoch上单一sample的平均loss与准确率
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects / len(dataloaders[phase].dataset)
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # 若验证集上效果更好，则更新记录的模型
        if phase == 'eval' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        # 记录每一轮验证后的准确率
        if phase == 'eval':
            val_acc_history.append(epoch_acc)

    # 完成训练，打印最优的验证准确率
    print('Best val Acc: {:4f}'.format(best_acc))
    # 加载最优模型的参数用于返回
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


if __name__ == '__main__':
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

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

    id2class = dict([(index, class_name) for index, class_name in enumerate(train_dataset.classes)])
    json.dump(id2class, open('./id2class.json', 'w'))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=20)
    loader = {'train': train_loader, 'eval': eval_loader}

    lenet = LeNet(len(train_dataset.classes))

    loss_func = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lenet.parameters(), lr=0.001)
    trained_model, _ = train_model(lenet, loader, loss_func, optim, 10)

    torch.save(trained_model.state_dict(), './lenet.pth')
