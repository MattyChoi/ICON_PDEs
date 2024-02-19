import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eig


# given a the number of steps, return a random bernoulli potential function 
def generate_random_potential_fn(pieces=5, ground=5.0):
    
    def bernoulli(x):
        # assert that all values given in x are between 0 and 1
        assert torch.all((0 <= x) & (x <= 1))

        # convert range [0, 1) to integer range [0, pieces-1]
        eps = 1e-6
        x = (torch.clamp(x, max=1.0-eps)* pieces).int()

        # create a random list of 0s and 1s representing the piecewise bernoulli function
        bern = torch.bernoulli(torch.rand(pieces))
        
        # add a ground potential energy
        # v_0 = np.random.rand(ground)

        # return 0s and 1s based on our random bernoulli list
        return bern[x] # + v_0

    # Generate a random nonnegative potential function
    return bernoulli


def finite_diff(grid, potential_fn):
    # get the number of steps, do not include the first and last points
    grid = grid[1:-1]
    steps = len(grid)

    # Construct the Hamiltonian matrix for the Schrödinger equation
    laplacian_matrix = torch.diag(-2 * torch.ones(steps)) \
                    + torch.diag(torch.ones(steps - 1), 1) \
                    + torch.diag(torch.ones(steps - 1), -1)
    
    kinetic_term = -laplacian_matrix / (steps + 1)**2
    potential_term = torch.diag(potential_fn(grid))
    hamiltonian = kinetic_term + potential_term

    return hamiltonian


def solve_schrodinger_eq(hamiltonian):
    # Solve the Schrödinger equation to find eigenvalues
    eigenvals, eigenvecs = torch.linalg.eig(hamiltonian)

    # sort from smallest to greatest eigenvalue
    inds = torch.argsort(eigenvals.real)

    return eigenvals[inds].real, eigenvecs[:, inds].real



# if __name__ == "__main__":
#     # Define the parameters
#     domain_size = 101
#     x_values = torch.linspace(0, 1, domain_size)
    
#     # Generate a random nonnegative potential function
#     potential = generate_random_potential_fn(pieces=50)

#     A = finite_diff(x_values, potential)
#     vals, vecs = solve_schrodinger_eq(A)
    
#     # Plot the potential and the first few eigenfunctions
#     plt.figure(figsize=(12,10))
#     for i in range(4):
#         y = []
#         y = np.append(y,vecs[:,i])
#         y = np.append(y,0)
#         y = np.insert(y,0,0)
#         plt.plot(x_values,y,lw=3, label="{} ".format(i))
#         plt.xlabel('x', size=14)
#         plt.ylabel('$\psi$(x)',size=14)
#     plt.legend()
#     plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
#     plt.show()