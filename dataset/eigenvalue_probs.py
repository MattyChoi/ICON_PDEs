import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset
from tfrecord.torch.dataset import TFRecordDataset


# create a pytorch dataset of eigenvalue problems
class EigvalueProbs(Dataset):
    """
    Dataset of eigenvalue problems for the Schrödinger equation
    """
    def __init__(self, data_dir="", transform=None):
        super(EigvalueProbs, self).__init__()

        self.data = TFRecordDataset("./data/finite_diff.tfrecord", index_path=None)


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)
    
