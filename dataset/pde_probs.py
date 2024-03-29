import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, default_collate
from tfrecord.torch.dataset import TFRecordDataset



# create a pytorch dataset of pde problems
class PDEProblems(Dataset):
    """
    Dataset of pde problems
    """
    def __init__(
        self, 
        data_dir="./data/finite_diff.tfrecord", 
        transform=None,
    ):
        super(PDEProblems, self).__init__()

        # load the iterable dataset from the tfrecord file
        ds = list(TFRecordDataset(data_dir, index_path=None))
        shuffle = torch.randperm(len(ds))
        
        self.data = []
            

    def __getitem__(self, index):
        # create a single prompt, using a number of examples from our dataset
        examples = self.data[index]
    

    def __len__(self):
        return len(self.data)
    

    def collate_fn(self, batch):
        inputs, labels = list(zip(*batch))

        inputs = torch.stack(inputs, dim=0)
        labels = torch.stack(labels, dim=0)

        return inputs, labels 
    
