import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, default_collate
from tfrecord.torch.dataset import TFRecordDataset


def one_hot_encode(index, num_examples, max_num_pairs):
    """
    One-hot encode an index
    """
    encoding = torch.zeros(num_examples, max_num_pairs)
    encoding[index] = 1

    return encoding


def trigonometric_encode(index, num_examples, max_num_pairs):
    """
    Create a trigonometric encoding of an index
    """
    encoding = torch.ones(max_num_pairs)
    half_dim = num_examples // 2
    emb = np.log(10000) / (half_dim)
    emb = torch.exp(torch.arange(half_dim + 1) * -emb)

    if index % 2 == 0:
        trig = emb[index // 2].sin()
    else:
        trig = emb[index // 2].cos()
    encoding *= trig

    return encoding


def create_prompts_only(examples, max_num_pairs=None, encoding="trig"):
    grid, conds, qois = examples['grid'], examples['conditions'], examples['qois']

    # get the number of examples given
    num_examples = len(grid)

    # get the number of positions in the grid
    num_pos = len(grid[0])

    # get the max number of condition/qoi pairs
    max_num_pairs = int(len(grid[0]) * 0.5) if not max_num_pairs else min(max_num_pairs, len(grid[0]))  

    terms = []
    positions = []
    values = []
    indices = []
    labels = []

    # term is 0 or 1 based on whether it is a condition/qoi value or boundary value respectively
    cond_term = torch.zeros(max_num_pairs)
    cond_term[-2:] = 1      # the last two columns will be reserved for the boundary values

    # iterate over the examples to create the prompt
    for i in range(num_examples):
        # randomly choose max_num_pairs of condition/qoi pairs
        inds = (torch.randperm(num_pos-2) + 1)[:max_num_pairs-2]   # leave out two for the boundary conditions
        boundary_inds = torch.tensor([0, -1])
        inds = torch.cat([inds, boundary_inds])

        # append the terms for the conditions
        terms.append(cond_term)

        # get the positions in the grid
        cond_pos = grid[i][inds]

        # append the positions for the conditions
        positions.append(cond_pos)

        # construct the values for the corresponding positions
        cond_val = conds[i][inds]
        qoi_val = qois[i][inds]

        # append the values of the conditions and qois
        values.append(cond_val)
        labels.append(qoi_val)

        # construct the index, use one-hot encoding or trigonometric encoding
        if encoding not in ["one-hot", "trig"]:
            raise ValueError(f"Invalid encoding: {encoding}")
        elif encoding == "one-hot":
            cond_ind = one_hot_encode(i, num_examples, max_num_pairs)
        elif encoding == "trig":
            cond_ind = trigonometric_encode(i, num_examples, max_num_pairs)

        # append the indices of the conditions and qois
        indices.append(cond_ind)

    # concatenate all the terms, positions, values and indices
    terms = torch.hstack(terms)
    positions = torch.hstack(positions)
    values = torch.hstack(values)
    indices = torch.hstack(indices)
    
    # stack the terms, positions, values and indices to create the prompt
    prompt = torch.vstack((terms, positions, values, indices)).t()

    # stack the labels
    labels = torch.hstack(labels)

    return prompt, torch.zeros(3,3), labels


def create_prompt_query(examples, max_num_pairs=None, encoding="trig"):
    grid, conds, qois = examples['grid'], examples['conditions'], examples['qois']

    # get the number of examples given
    num_examples = len(grid)

    # get the number of positions in the grid
    num_pos = len(grid[0])

    # get the max number of condition/qoi pairs
    max_num_pairs = int(len(grid[0]) * 0.5) if not max_num_pairs else min(max_num_pairs, len(grid[0]))  

    terms = []
    positions = []
    values = []
    indices = []

    # term is 0 or 1 based on whether it is a condition/qoi value or boundary value respectively
    cond_term = torch.zeros(max_num_pairs)
    qoi_term = torch.zeros(max_num_pairs)
    cond_term[-2:] = 1      # the last two columns will be reserved for the boundary values

    # iterate over the examples except the last one to create the prompt
    for i in range(num_examples-1):
        # randomly choose max_num_pairs of condition/qoi pairs
        cond_inds = (torch.randperm(num_pos-2) + 1)[:max_num_pairs-2]   # leave out two for the boundary conditions
        boundary_inds = torch.tensor([0, -1])
        cond_inds = torch.cat([cond_inds, boundary_inds])
        qoi_inds = (torch.randperm(num_pos-2) + 1)[:max_num_pairs]

        # append the terms for the conditions and qois
        terms.append(cond_term)
        terms.append(qoi_term)

        # get the positions in the grid
        cond_pos = grid[i][cond_inds]
        qoi_pos = grid[i][qoi_inds]

        # append the positions for the conditions and qois
        positions.append(cond_pos)
        positions.append(qoi_pos)

        # construct the values for the corresponding positions
        cond_val = conds[i][cond_inds]
        qoi_val = qois[i][qoi_inds]

        # append the values of the conditions and qois
        values.append(cond_val)
        values.append(qoi_val)

        # construct the index, use one-hot encoding or trigonometric encoding
        if encoding not in ["one-hot", "trig"]:
            raise ValueError(f"Invalid encoding: {encoding}")
        elif encoding == "one-hot":
            cond_ind = one_hot_encode(i, num_examples, max_num_pairs)
            qoi_ind = -cond_ind
        elif encoding == "trig":
            cond_ind = trigonometric_encode(i, num_examples, max_num_pairs)
            qoi_ind = -cond_ind

        # append the indices of the conditions and qois
        indices.append(cond_ind)
        indices.append(qoi_ind)
        
    # use the last example to create the question condition, the query key, and the labels
    # randomly choose max_num_pairs of condition/qoi pairs
    cond_inds = (torch.randperm(num_pos-2) + 1)[:max_num_pairs-2]   # leave out two for the boundary conditions
    boundary_inds = torch.tensor([0, -1])
    cond_inds = torch.cat([cond_inds, boundary_inds])
    qoi_inds = (torch.randperm(num_pos-2) + 1)[:max_num_pairs]

    # append the terms for the question conditions
    terms.append(cond_term)

    # get the positions in the grid
    cond_pos = grid[-1][cond_inds]

    # append the positions for the question conditions
    positions.append(cond_pos)

    # construct the values for the corresponding positions
    cond_val = conds[-1][cond_inds]

    # append the values of the questions conditions
    values.append(cond_val)

    # construct the index, use one-hot encoding or trigonometric encoding
    if encoding not in ["one-hot", "trig"]:
        raise ValueError(f"Invalid encoding: {encoding}")
    elif encoding == "one-hot":
        cond_ind = one_hot_encode(num_examples-1, num_examples, max_num_pairs)
    elif encoding == "trig":
        cond_ind = trigonometric_encode(num_examples-1, num_examples, max_num_pairs)

    # append the indices of the conditions and qois
    indices.append(cond_ind)

    # create the query key
    query_term = torch.zeros(max_num_pairs)
    query_pos = grid[-1][qoi_inds]
    query = torch.vstack([query_term, query_pos]).t()

    # create the labels
    labels = qois[-1][qoi_inds]

    # concatenate all the terms, positions, values and indices
    terms = torch.hstack(terms)
    positions = torch.hstack(positions)
    values = torch.hstack(values)
    indices = torch.hstack(indices)
    
    # stack the terms, positions, values and indices to create the prompt
    prompt = torch.vstack((terms, positions, values, indices)).t()

    return prompt, query, labels


# create a pytorch dataset of eigenvalue problems
class PDEProblems(Dataset):
    """
    Dataset of pde problems for the Schrödinger equation
    """
    def __init__(
        self, 
        data_dir="./data/finite_diff.tfrecord", 
        num_examples=5, 
        max_num_pairs=None,
        encoding="trig",
        prompts_only=False,
        transform=None,
    ):
        super(PDEProblems, self).__init__()

        self.num_examples = num_examples
        self.max_num_pairs = max_num_pairs
        self.encoding = encoding
        self.prompts_only = prompts_only

        # load the iterable dataset from the tfrecord file
        ds = list(TFRecordDataset(data_dir, index_path=None))
        shuffle = torch.randperm(len(ds))

        # create a list of examples, each containing a number of examples
        self.data = []
        for i in range(len(ds) // num_examples):
            self.data.append([])
            for j in range(num_examples):
                self.data[i].append(ds[shuffle[i*num_examples+j]])
            

    def __getitem__(self, index):
        # create a single prompt, using a number of examples from our dataset
        examples = self.data[index]
        examples = default_collate(examples)

        if self.prompts_only:
            return create_prompts_only(
                examples, 
                max_num_pairs=self.max_num_pairs, 
                encoding=self.encoding,
            )
        return create_prompt_query(
            examples, 
            max_num_pairs=self.max_num_pairs, 
            encoding=self.encoding,
        )
    

    def __len__(self):
        return len(self.data)
    

    def collate_fn(self, batch):
        prompts, queries, labels = list(zip(*batch))

        prompts = torch.stack(prompts, dim=0)
        queries = torch.stack(queries, dim=0)
        labels = torch.stack(labels, dim=0)

        return prompts, queries, labels 
    

# create a pytorch dataset of eigenvalue problems
class EigvalueProbsIter(IterableDataset):
    """
    Dataset of eigenvalue problems for the Schrödinger equation
    """
    def __init__(
        self, 
        data_dir="./data/finite_diff.tfrecord", 
        num_examples=5, 
        max_num_pairs=None,
        encoding="trig",
        transform=None
    ):
        super(EigvalueProbs, self).__init__()

        self.num_examples = num_examples
        self.max_num_pairs = max_num_pairs
        self.encoding = encoding

        # load the iterable dataset from the tfrecord file
        self.data = iter(TFRecordDataset(data_dir, index_path=None))


    def __iter__(self):
        # create a single prompt, using a number of examples from our dataset
        examples = [next(self.data) for _ in range(self.num_examples)]
        examples = default_collate(examples)

        yield create_prompt_query(examples, max_num_pairs=self.max_num_pairs, encoding=self.encoding)

        
    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)


# if __name__ == "__main__":
#     data = EigvalueProbs(prompts_only=True)
    
#     prompt, query, labels = data[0]
    
#     from utils.visualizations import plot_ground_state
#     plot_ground_state(prompt[:, 1], labels)