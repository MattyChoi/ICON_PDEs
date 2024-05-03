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
        data_dir="./data/finite_diff_gsize_101.tfrecord", 
        transform=None,
    ):
        super(PDEProblems, self).__init__()

        # load the iterable dataset from the tfrecord file
        ds = list(TFRecordDataset(data_dir, index_path=None))
        shuffle = torch.randperm(len(ds))
        
        # create a list of operator examples and their corresponding QoIs and 
        self.data = []
        for i in shuffle:
            example = ds[i]
            self.data.append(example)
            

    def __getitem__(self, index):
        # create a single prompt, using a set of examples from one operator from our dataset
        examples = self.data[index]
        grid, conds, qois = examples['grid'], examples['conditions'], examples['qois']
        grid, conds, qois = torch.Tensor(grid), torch.Tensor(conds), torch.Tensor(qois)
        
        # get the number of examples and grid_size
        grid_size = grid.size(0)
        num_examples = conds.size(0) // grid_size 
        
        # reshape the conditions and qois back to their 2d array forms
        conds = conds.reshape(num_examples, grid_size)
        qois = qois.reshape(num_examples, grid_size)

        # create the label
        label = torch.hstack([conds, qois])
        
        # shuffle the label so that we are predicting different qois each time
        shuffle = torch.randperm(num_examples)
        label = label[shuffle]
        
        prompt = label.clone().detach()
        assert grid_size == prompt.size(1) // 2
        prompt[-1, grid_size:] = 0
        
        return prompt, label

    def __len__(self):
        return len(self.data)
    

    def collate_fn(self, batch):
        prompts, labels = list(zip(*batch))

        prompts = torch.stack(prompts, dim=0)
        labels = torch.stack(labels, dim=0)

        return prompts, labels 
    


# if __name__ == "__main__":
#     data = PDEProblems()
    
#     index = 2
#     prompt, label = data[index]