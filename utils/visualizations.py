import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ground_state(conditions, qois):
    # Plot the domain with the eigenfunction
    plt.figure(figsize=(12,10))
    plt.plot(conditions.numpy(), qois.numpy(),lw=3, label="{} ".format(i))
    plt.xlabel('x', size=14)
    plt.ylabel('$\psi$(x)',size=14)
    plt.legend()
    plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)
    plt.show()