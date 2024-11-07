import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from model import *

dataset_transform = transforms.Compose([
    transforms.ToTensor()
])
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,download=True,transform=dataset_transform)
test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=False)
img,target = test_set[0]

print("img.shape：",img.shape)
print("target：",target)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs,targets = data
        writer.add_images("Epoch:{}".format(epoch),imgs,step)
        step += 1
writer.close()