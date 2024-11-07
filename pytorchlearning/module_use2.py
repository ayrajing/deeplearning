import torchvision
from PIL import Image
import torch
import torch.nn as nn
image_path = "./dog.jpeg"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                            torchvision.transforms.ToTensor()
                                           ])
image = transform(image)
print(image)
model = torch.load("tudui_0.pth",map_location=torch.device("cpu"))
print(model)
image = torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    image = image.to("cpu")
    output = model(image)
print(output)
print(output.argmax(1).item())

# 添加类别映射
classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 在预测部分
with torch.no_grad():
    output = model(image)
    predicted_class = output.argmax(1).item()
    print(f"预测类别: {classes[predicted_class]}")

