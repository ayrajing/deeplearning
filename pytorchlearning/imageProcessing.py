from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
img_path = "/Users/jingjing/Documents/cursor_zwj/pytorchlearning/photo.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs") 
trans_random = transforms.RandomCrop(30)
trans_totensor = transforms.ToTensor() 
trans_compose_2 = transforms.Compose([trans_random,trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop",img_crop,i)  
writer.close()
