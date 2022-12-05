"""
常规的训练过程使用的方法。包括：
create_loader: 对于适用于ImageFolder的文件结构的数据，生成dataloader
train: 执行训练、验证过程，保存最优的模型参数
"""

import torch
import copy
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def create_loader(root_dir, transformer, batch_size, shuffle=True, num_workers=0):
    train_dataset = ImageFolder(root=os.path.join(root_dir, 'train'),
                                transform=transformer['train'])
    eval_dataset = ImageFolder(root=os.path.join(root_dir, 'eval'),
                               transform=transformer['eval'])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, eval_loader


def train(model, dataloaders, criterion, optimizer, device, num_epochs, is_inception=False, inception_weight=0.4):
    """
    通用的模型训练方式
    :param model: 实例化的nn.Module模型对象
    :param dataloaders: dict，形如{'train': train_loader, 'eval': eval_loader}
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 使用的设备 torch.device('cuda') or torch.device('cpu')
    :param num_epochs: 训练轮数
    :param is_inception: 是否添加inception。用于googlenet
    :param inception_weight: inception辅助分类器的loss权重
    :return: 完成训练的模型，以及每一轮在验证集上的准确率
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
                    # 若带有inception模式且在训练中，则分别计算outputs与aux_outputs的loss：
                    if is_inception is True and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = .0
                        for aux_output in aux_outputs:
                            loss2 += criterion(aux_output, labels)
                        loss = loss1 + inception_weight * loss2
                    # 否则，直接计算loss
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # 训练模式下，更新梯度
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss
                running_corrects += torch.sum(preds == labels)

                # 在训练中，每隔len(dataloader) // 10次，打印一次该epoch上单一sample的平均loss与准确率
                if phase == 'train':
                    every = len(dataloaders['train']) // 10
                    if index % every == 0:
                        tmp_loss = running_loss / ((index + 1) * dataloaders['train'].batch_size)
                        tmp_acc = running_corrects / ((index + 1) * dataloaders['train'].batch_size)
                        print('Batches {}/{}, Loss: {}, Acc: {}'.format(
                            index, len(dataloaders['train']), round(tmp_loss.item(), 6), round(tmp_acc.item(), 6)))

            # 每个epoch训练/验证完毕后，统计该epoch上单一sample的平均loss与准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))

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
