import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.finite_diff import solve_schrodinger_eq


# given a the number of steps, return a random bernoulli potential function 
def generate_random_potential_fn(pieces=5, ground=5.0):
    
    def bernoulli(x):
        # assert that all values given in x are between 0 and 1
        assert np.all((0 <= x) & (x <= 1))

        x = np.clip(x, 0, 1 - 1e-10)

        # create a random list of 0s and 1s representing the piecewise bernoulli function
        bern = np.random.choice([0, 1], size=pieces)
        
        # add a ground potential energy
        # v_0 = np.random.rand(ground)

        # return 0s and 1s based on our random bernoulli list
        return bern[(x * pieces).astype(int)] # + v_0

    # Generate a random nonnegative potential function
    return bernoulli


# create a pytorch dataset of eigenvalue problems
class EigvalueProbs(Dataset):
    """
    Dataset of eigenvalue problems for the Schrödinger equation
    """
    def __init__(self, transform=None):
        super(EigvalueProbs, self).__init__()

        self.data = []


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


    def collate_fn(self, batch):
        return torch.stack(batch, dim=0)
    

if __name__ == "__main__":
    # Define the parameters
    domain_size = 100
    x_values = np.linspace(0, 1, domain_size)
    
    # Generate a random nonnegative potential function
    potential = generate_random_potential_fn()
    print(potential(x_values))
