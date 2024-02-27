import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_ground_state(conditions, qois, labels=None, show=True):
    if isinstance(conditions, torch.Tensor):
        conditions = conditions.cpu().detach().numpy()
    if isinstance(qois, torch.Tensor):  
        qois = qois.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):  
        labels = labels.cpu().detach().numpy()

    # sort the data
    sort_inds = np.argsort(conditions)
    conditions = conditions[sort_inds]
    qois = qois[sort_inds]
    if labels is not None:
        labels = labels[sort_inds]

    # Plot the domain with the eigenfunction
    fig = plt.figure(figsize=(12,10))
    plt.plot(conditions, qois, lw=3, label="preds")
    if labels is not None:
        plt.plot(conditions, labels, lw=3, label="truth")
    plt.xlabel('x', size=14)
    plt.ylabel('$\psi$(x)',size=14)
    plt.legend()
    plt.title('normalized wavefunctions for a harmonic oscillator using finite difference method',size=14)

    if show:
        plt.show()

    return fig