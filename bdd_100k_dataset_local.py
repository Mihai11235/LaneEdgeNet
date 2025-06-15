from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


class BDD100KDatasetLocal(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = os.listdir(images_dir)
        self.mask_resize = transforms.Resize(mask_size)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = 1 - mask
        mask = (mask > 0.1).int().float()
        
        return image, mask