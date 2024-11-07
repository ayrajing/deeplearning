from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms
import torch

writer = SummaryWriter("logs") # 日志文件存储位置
img_path = "/Users/jingjing/data/MNIST/raw/pytorchlearning/photo.jpg"
img = Image.open(img_path)

# 检查图像模式
print(f"图像模式: {img.mode}")
# 将图像转换为张量
tensor_trans = transforms.ToTensor() 
tensor_img = tensor_trans(img)
# 打印张量形状和类型
print(f"张量形状: {tensor_img.shape}")
print(f"张量类型: {tensor_img.dtype}")
# 根据图像通道数调整标准化参数s
if tensor_img.shape[0] == 1:
    trans_norm = transforms.Normalize([0.5], [0.5])
elif tensor_img.shape[0] == 3:
    trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
else:
    raise ValueError(f"Unexpected number of channels: {tensor_img.shape[0]}")

img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
# 打印归一化后的张量：这表示标准化后图像张量的第一个通道（红色）中，第一行第一列的像素值。
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)
writer.close()
