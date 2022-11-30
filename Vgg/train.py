import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

import copy

from model_structure import Vgg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloaders, criterion, optimizer, num_epochs):
    val_acc_history = list()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            elif phase == 'eval':
                model.eval()

            running_loss, running_corrects = .0, 0
            for index, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs.to(device)
                labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()