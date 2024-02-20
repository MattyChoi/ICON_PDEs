import os
import sys
# Import modules from base directory
sys.path.insert(0, os.getcwd())

import argparse
import tensorflow as tf
import torch
import numpy as np
from tqdm import tqdm

from dataset.finite_diff import generate_random_potential_fn, finite_diff, get_eigvals_and_eigvecs


def convert_to_tf_example(data_dic):
    # create a dictionary of tf.train.Features of our conditions and qois
    features = {}
    for key, value in data_dic.items():
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        elif isinstance(value, int):
            value = np.array([value])

        # Create a tf.train.Feature for each NumPy array
        features[key] = tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
        
    # Create a tf.train.Features containing all the features
    features = tf.train.Features(
        feature=features
    )

    # Create a tf.train.Example
    example = tf.train.Example(features=features)

    return example.SerializeToString()


def finite_diff_dataset(path: str, num_operators: int, gridsize: int):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, "finite_diff.tfrecord")

    # Create a tfrecord writer and overwrite any existing file
    with tf.io.TFRecordWriter(file_path) as writer:
        # Create a grid
        grid = torch.linspace(0, 1, gridsize)

        # Create a dataset of eigenvalue problems
        for i in tqdm(range(num_operators)):
            # Generate a random potential function
            potential_fn = generate_random_potential_fn(pieces=10)

            # Create a Hamiltonian matrix
            hamiltonian = finite_diff(grid[1:-1], potential_fn)

            # Get the eigenvalues and eigenvectors
            _, eigenvecs = get_eigvals_and_eigvecs(hamiltonian)
            
            # Convert PyTorch tensors to NumPy arrays and store in dictionary
            data_dic = {
                "grid": grid.numpy(),
                "conditions": potential_fn(grid).numpy(),
                "qois": np.pad(eigenvecs[:,0].numpy(), (1, 1)),
            }

            # Create an example
            example = convert_to_tf_example(data_dic)

            # Write the example
            writer.write(example)



def main():
    parser = argparse.ArgumentParser(description='Creates a tfrecord dataset of eigenvalue problems for the Schrödinger equation.')

    # Add command-line arguments or flags
    parser.add_argument('-n', '--number', type=int, default=10000, help='number of operators to generate')
    parser.add_argument('-g', '--gridsize', type=int, default=101, help='size of the grid sampled for each operator')
    parser.add_argument('-p', '--path', type=str, default="./data", help='folder path to save the tfrecord file')

    args = parser.parse_args()

    finite_diff_dataset(args.path, args.number, args.gridsize)


if __name__ == "__main__":
    main()