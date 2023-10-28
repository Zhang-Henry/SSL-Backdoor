import os, torch
from torch.utils import data
from PIL import Image
import numpy as np

class FileListDataset(data.Dataset):
    def __init__(self, path_to_txt_file, transform):
        # self.data_root = data_root
        with open(path_to_txt_file, 'r') as f:
            self.file_list = f.readlines()
            self.file_list = [row.rstrip() for row in self.file_list]

        self.transform = transform


    def __getitem__(self, idx):
        image_path, label = self.file_list[idx].split()
        img = Image.open(image_path).convert('RGB')

        img= self.transform(img)

        return img,torch.tensor(int(label))

    def __len__(self):
        return len(self.file_list)
