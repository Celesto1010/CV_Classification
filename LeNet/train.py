import torch
import torch.nn as nn
from model_structure import LeNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from train_procedure import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = datasets.CIFAR10(
        root=r'E:\PythonProjects\CV\Datasets\CIFAR10\train',
        train=True,
        download=False,
        transform=transformer
    )
    eval_dataset = datasets.CIFAR10(
        root=r'E:\PythonProjects\CV\Datasets\CIFAR10\eval',
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
    trained_model, _ = train(lenet, loader, loss_func, optim, device, 10)

    torch.save(trained_model.state_dict(), './lenet.pth')
