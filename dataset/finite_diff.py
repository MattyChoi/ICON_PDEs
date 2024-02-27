import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
import matplotlib.pyplot as plt


# given a the number of steps, return a random bernoulli potential function 
def generate_random_potential_fn(pieces=8, ground=5.0):
    
    def bernoulli(x):
        # assert that all values given in x are between 0 and 1
        assert torch.all((0 <= x) & (x <= 1))

        # convert range [0, 1) to integer range [0, pieces-1]
        eps = 1e-6
        x = (torch.clamp(x, max=1.0-eps)* pieces).int()

        # create a random list of 0s and 1s representing the piecewise bernoulli function
        bern = torch.bernoulli(torch.rand(pieces))
        
        # add a ground potential energy
        v_0 = torch.rand(1) * ground

        # return 0s and 1s based on our random bernoulli list
        return bern[x] + v_0

    # Generate a random nonnegative potential function
    return bernoulli


def finite_diff(grid, potential_fn):
    dim = len(grid)

    # Construct the Hamiltonian matrix for the Schrödinger equation
    laplacian_matrix = torch.diag(-2 * torch.ones(dim)) \
                    + torch.diag(torch.ones(dim - 1), 1) \
                    + torch.diag(torch.ones(dim - 1), -1)
    
    kinetic_term = -laplacian_matrix / (grid[1] - grid[0])**2
    potential_term = torch.diag(potential_fn(grid))
    hamiltonian = kinetic_term + potential_term

    return hamiltonian


def get_eigvals_and_eigvecs(hamiltonian):
    eigenvals, eigenvecs = torch.linalg.eigh(hamiltonian)

    # sort from smallest to greatest eigenvalue
    inds = torch.argsort(eigenvals.real)
    eigenvals, eigenvecs = eigenvals[inds].real, eigenvecs[:, inds].real
    assert eigenvals[0] > 0
    if eigenvecs[:, 0].min() >= 0:
        eigenvecs[:, 0] *= -1

    return eigenvals, eigenvecs



if __name__ == "__main__":
    # Define the parameters
    domain_size = 101
    x_values = torch.linspace(0, 1, domain_size)[1:-1]
    
    # Generate a random nonnegative potential function
    potential = generate_random_potential_fn(pieces=10, ground=5.0)

    A = finite_diff(x_values, potential)
    vals, vecs = get_eigvals_and_eigvecs(A)
    
    # Plot the potential and the first few eigenfunctions
    plt.figure(figsize=(12,10))
    # plt.plot(x_values,potential(x_values),lw=3)
    for i in range(1):
        plt.plot(x_values,vecs[:,i].numpy(),lw=3, label="{} ".format(i))
        plt.xlabel('x', size=14)
        plt.ylabel('$\psi$(x)',size=14)
        print(vecs[:,i].norm())
        print(vecs[:,i].max())
        print(vecs[:,i].min())
    plt.legend()
    plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
    plt.show()