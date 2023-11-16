import os, torch
from torch.utils import data
from PIL import Image
import numpy as np

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, clean_transform, bd_transform):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.clean_transform = clean_transform
        self.bd_transform = bd_transform


    def __getitem__(self, idx):
        image_path, label = self.file_list[idx].split()
        img = Image.open(image_path).convert('RGB')

        img_clean = self.clean_transform(img)
        img_bd = self.bd_transform(img)

        return img_clean,img_bd,torch.tensor(int(label))

    def __len__(self):
        return len(self.file_list)
