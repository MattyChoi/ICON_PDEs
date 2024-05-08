import matplotlib.pyplot as plt
import numpy as np
import torch


def closest_factors(x):
    fc = int(x**0.5)
    while (x % fc > 0):
        fc -= 1
    
    return x // fc, fc


def plot_figure(conditions, qois, labels=None, inds=None, show=True):
    if isinstance(conditions, torch.Tensor):
        conditions = conditions.cpu().detach().numpy()
    if isinstance(qois, torch.Tensor):  
        qois = qois.cpu().detach().numpy()
    if isinstance(labels, torch.Tensor):  
        labels = labels.cpu().detach().numpy()

    # # sort the data
    # sort_inds = np.argsort(conditions)
    # conditions = conditions[sort_inds]
    # qois = qois[sort_inds]
    # if labels is not None:
    #     labels = labels[sort_inds]

    # Plot the domain with the eigenfunction
    fig = plt.figure(figsize=(12,12))
    if inds is not None:
        num_sub_plots = len(inds)
        rows, cols = closest_factors(num_sub_plots)
        for i, ind in enumerate(inds):
            plt.subplot(rows, cols, i + 1)
            
            plt.plot(conditions, qois[ind], lw=3, label="preds")
            if labels is not None:
                plt.plot(conditions, labels[ind], lw=3, label="truth")
            plt.xlabel('x', size=14)
            plt.ylabel('$\psi$(x)',size=14)
            plt.legend()
    else:    
        plt.plot(conditions, qois, lw=3, label="preds")
        if labels is not None:
            plt.plot(conditions, labels, lw=3, label="truth")
        plt.xlabel('x', size=14)
        plt.ylabel('$\psi$(x)',size=14)
        plt.legend()
        # plt.title('',size=14)

    if show:
        plt.show()

    return fig