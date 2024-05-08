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


def finite_diff_dataset_eigvals(path: str, num_operators: int, gridsize: int):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, "finite_diff_eigvals") + f"_gsize_{gridsize}" + ".tfrecord" 

    # Create a tfrecord writer and overwrite any existing file
    with tf.io.TFRecordWriter(file_path) as writer:
        # Create a grid
        grid = torch.linspace(0, 1, gridsize)

        # Create a dataset of eigenvalue problems
        for _ in tqdm(range(num_operators)):
            # Generate a random potential function
            potential_fn = generate_random_potential_fn(pieces=10)

            # Create a Hamiltonian matrix
            hamiltonian = finite_diff(grid[1:-1], potential_fn)

            # Get the eigenvalues and eigenvectors
            _, eigenvecs = get_eigvals_and_eigvecs(hamiltonian)
            conditions = potential_fn(grid).numpy()
            qois = eigenvecs[:,0].numpy()
            
            # Convert PyTorch tensors to NumPy arrays and store in dictionary
            data_dic = {
                "grid": grid.numpy(),
                "conditions": conditions,
                "qois": np.pad(qois.numpy(), (1, 1)),
            }

            # Create an example
            example = convert_to_tf_example(data_dic)

            # Write the example
            writer.write(example)


def finite_diff_dataset(path: str, num_operators: int, num_examples: int, gridsize: int):
    if not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, f"gsize_{gridsize}_numex_{num_examples}_numop_{num_operators}.tfrecord")

    # Create a tfrecord writer and overwrite any existing file
    with tf.io.TFRecordWriter(file_path) as writer:
        # Create a grid
        grid = torch.linspace(0, 1, gridsize)

        # Create a dataset of different pde operators
        for _ in tqdm(range(num_operators)):
            # Generate a random potential function
            potential_fn = generate_random_potential_fn(pieces=10)

            # Create a Hamiltonian matrix
            hamiltonian = finite_diff(grid[1:-1], potential_fn)
            
            # Create a number of examples for this specific operator
            # Each prompt will have n condition and qoi pairs (RHS solutions)
            conditions = []
            qois = []
            for _ in range(num_examples):
                f = torch.randn(hamiltonian.size(0))
                conditions.append(np.pad(f.numpy(), (1, 1)))
                qois.append(np.pad(torch.linalg.pinv(hamiltonian) @ f, (1, 1)))
            
            # Convert PyTorch tensors to NumPy arrays and store in dictionary
            data_dic = {
                "grid": grid.numpy(),
                "conditions": np.hstack(conditions),
                "qois": np.hstack(qois),
            }

            # Create an example
            example = convert_to_tf_example(data_dic)

            # Write the example
            writer.write(example)



def main():
    parser = argparse.ArgumentParser(description='Creates a tfrecord dataset of eigenvalue problems for the Schrödinger equation.')

    # Add command-line arguments or flags
    parser.add_argument('-o', '--numop', type=int, default=1000, help='number of operators to generate')
    parser.add_argument('-n', '--numex', type=int, default=1000, help='number of examples to generate for each operator')
    parser.add_argument('-g', '--gridsize', type=int, default=101, help='size of the grid sampled for each operator')
    parser.add_argument('-p', '--path', type=str, default="./data", help='folder path to save the tfrecord file')
    parser.add_argument('-e', '--eigen', type=bool, default=False, help='get ground state eigenfunction as qoi')

    args = parser.parse_args()

    if args.eigen:
        finite_diff_dataset_eigvals(args.path, args.numop, args.gridsize)
    else:
        finite_diff_dataset(args.path, args.numop, args.numex, args.gridsize)


if __name__ == "__main__":
    main()