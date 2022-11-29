"""
预测脚本
"""

import json
import torch
from torchvision import transforms
from PIL import Image
from model_structure import LeNet


transformer = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


model = LeNet(10)
model.load_state_dict(torch.load('./lenet.pth'))

image = Image.open('./plane2.jpeg')
image = transformer(image).to(torch.float)
image_tensor = torch.unsqueeze(image, 0)

with torch.no_grad():
    outputs = model(image_tensor)
    logits = torch.softmax(outputs, 1)
    prob, predict = torch.max(logits, 1)

id2class = json.load(open('./id2class.json', 'r'))
print('分类结果为：{0}，可能性为：{1}'.format(id2class[str(predict.item())], prob.item()))


