"""
Alexnet模型训练
"""
import torch
import torch.nn as nn
from model_structure import AlexNet
from torchvision import transforms
import json
from train_procedure import train, create_loader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 若用的是使用imagenet预训练的模型，则Normalize的mean为（123.68, 116.78, 103.94）.
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

root_dir = r'E:\CV\Datasets\flower_photos_for_train'
train_loader, eval_loader = create_loader(root_dir, transformer=transformer, batch_size=32)
loader = {'train': train_loader, 'eval': eval_loader}

id2class = dict([(index, class_name) for index, class_name in enumerate(train_loader.dataset.classes)])
json.dump(id2class, open('./id2class.json', 'w'))

alexnet = AlexNet(num_classes=len(id2class))
alexnet.to(device)
loss_func = nn.CrossEntropyLoss()
optim = torch.optim.Adam(alexnet.parameters(), lr=0.0002)
trained_model, _ = train(alexnet, loader, loss_func, optim, device, 10)

torch.save(trained_model.state_dict(), './alexnet.pth')