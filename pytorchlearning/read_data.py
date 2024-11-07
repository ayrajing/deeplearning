from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)
    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir ="dataset/train"
ant_label_dir ="ants"
ants_dataset = MyData(root_dir, ant_label_dir)
img,label = ants_dataset[0]
#img.show()
bee_label_dir ="bees"
bees_dataset = MyData(root_dir, bee_label_dir)
img,label = bees_dataset[0]
#img.show()
train_dataset = ants_dataset + bees_dataset
print(len(ants_dataset))
print(len(bees_dataset))
print(len(train_dataset))
