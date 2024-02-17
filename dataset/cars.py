import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import torchvision
import torchvision.transforms as tsfm
from torch.utils.data import Dataset


class Cars(Dataset):
    """
    Dataset of car images
    """
    def __init__(self, data_dir="", transform=None):
        super(Cars, self).__init__()

        train = torchvision.datasets.StanfordCars(root=data_dir, transform=transform)
        test = torchvision.datasets.StanfordCars(root=data_dir, split="test", transform=transform)

        self.data = torch.utils.data.ConcatDataset([train, test])


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)

