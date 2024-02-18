import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh, eig



def df_dt(f, dt):
    '''
    @parameters:
        f: [nx, nt]
        dt: float
    @return:
        df_dt: [nx, nt-1]
    '''
    return (f[:,1:] - f[:,:-1]) / dt

def df_dx(f, dx, periodic = False):
    '''
    @parameters:
        f: [nx, nt]
        dx: float
        periodic: bool
    @return:
        df_dx: [nx, nt] (if not periodic copy the same value at the boundary)
    '''
    if periodic:
        ret = (jnp.roll(f, -1, axis = 0) - jnp.roll(f, 1, axis = 0)) / (2*dx)
    else:
        ret = (f[2:,:] - f[:-2,:]) / (2*dx)
        ret = jnp.concatenate([ret[0:1,:], ret, ret[-1:,:]], axis = 0)
    return ret

def d2f_dx2(f, dx, periodic = False):
    '''
    @parameters:
        f: [nx, nt]
        dx: float
        periodic: bool
    @return:
        d2f_dx2: [nx, nt] (if not periodic pad zero at the boundary)
    '''
    if periodic:
        ret = (jnp.roll(f, -1, axis = 0) - 2*f + jnp.roll(f, 1, axis = 0)) / (dx**2)
    else:
        ret = (f[2:,:] - 2*f[1:-1,:] + f[:-2,:]) / (dx**2)
        ret = jnp.pad(ret, ((1,1),(0,0)), 'constant', constant_values = 0.0)
    return ret



def solve_schrodinger_eq(hamiltonian):
    # Solve the Schrödinger equation to find eigenvalues
    eigenvalues, eigenvectors = eig(hamiltonian)
    inds = np.argsort(np.real(eigenvalues))
    return eigenvalues[inds], eigenvectors[:, inds]

def main():
    # Define the parameters
    domain_size = 100
    x_values = np.linspace(-5, 5, domain_size)
    
    # Generate a random nonnegative potential function
    potential = generate_random_potential(x_values)

    # Construct the Hamiltonian matrix for the Schrödinger equation
    h_bar = 1.0  # Planck's constant / (2 * pi)
    mass = 1.0   # Particle mass
    laplacian_matrix = np.diag(-2 * np.ones(domain_size)) + np.diag(np.ones(domain_size - 1), 1) + np.diag(np.ones(domain_size - 1), -1)
    kinetic_term = -(h_bar**2) / (2 * mass) * laplacian_matrix
    potential_term = np.diag(potential)
    hamiltonian = kinetic_term + potential_term



if __name__ == "__main__":
    main()