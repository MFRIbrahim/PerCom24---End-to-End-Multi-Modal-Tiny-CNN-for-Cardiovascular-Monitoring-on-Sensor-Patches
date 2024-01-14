import os
import torch
from torch.utils.data import Dataset
import numpy as np


class HybridDatasetnpy(Dataset):
    def __init__(self, input_dir, input_dir2, transform=None, transform2=None, target_transform=None):
        self.input_dir = input_dir 
        self.input_dir2 = input_dir2
        self.transform = transform
        self.transform2 = transform2
        self.target_transform = target_transform
        self.input_list = sorted(os.listdir(self.input_dir))
        self.input_list2 = sorted(os.listdir(self.input_dir2))

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx): 
        input_path = os.path.join(self.input_dir, f"{self.input_list[idx]}")
        input = torch.from_numpy(np.load(input_path)).type(torch.float32)  
        input = input.unsqueeze(0)
        input = input.unsqueeze(0)
        word = self.input_list[idx].split(".")[0]
        parts = word.split("_")
        record_number = parts[0]
        label = parts[1][-1]
        input_path2 = os.path.join(self.input_dir2, f"{self.input_list2[idx]}")
        input2 = torch.from_numpy(np.load(input_path2)).type(torch.float32)  
        input2 = input2.unsqueeze(0)
        input2 = input2.unsqueeze(0)
        if label == "N":
            label = torch.tensor([1,0], dtype=torch.float32)
        if label == "A":
            label = torch.tensor([0,1], dtype=torch.float32)
        if self.transform:
            input = self.transform(input)
        if self.transform2:
            input2 = self.transform2(input2)
        if self.target_transform:
            label = self.target_transform(label)
        return input, input2, label, int(record_number)