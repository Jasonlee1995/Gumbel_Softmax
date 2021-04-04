from PIL import Image, ImageFile

import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MNIST_dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.files = [self.data_dir + file for file in sorted(os.listdir(self.data_dir))]

    def __getitem__(self, idx):
        img = self.files[idx]
        img = Image.open(img).convert('L')
        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):
        return len(self.files)