import json
import torch
from torchvision import transforms
from PIL import Image
from model_structure import GoogLeNet


transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


image = Image.open('tulip2.jpeg')
image = transformer(image).to(torch.float)
image_tensor = torch.unsqueeze(image, 0)

model = GoogLeNet(num_classes=5, aux_logits=False)
model.load_state_dict(torch.load('./googlenet.pth'), strict=False)
model.eval()

with torch.no_grad():
    outputs = model(image_tensor)
    logits = torch.softmax(outputs, 1)
    prob, predict = torch.max(logits, 1)

id2class = json.load(open('./id2class.json', 'r'))
print('分类结果为：{0}，可能性为：{1}'.format(id2class[str(predict.item())], prob.item()))