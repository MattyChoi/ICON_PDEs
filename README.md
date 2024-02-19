# In-Context Operator Networks for Sturm-Liouville problems

This repo contains an implementation of ICON (In-Context Operator Networks) on the Sturm-Liouville problem to find the ground state eigenvectors (eigenvectors corresponding to the smallest eigenvalue). The conditions and QoIs (quantities of interest) will be the potential function $q(x)$ and $\lambda_0$, respectively with the $p$ and $w$ being the identity function

Once in the directory containing the contents of the repository, run
```
pip install -r requirements.txt
```
if you do not have all the necessary packages listed. Or, if you are using conda, you use the commands

```
conda env create --file environment.yml
conda activate icon
```

and 

```
conda env export > environment.yml 
```

and 

```
conda list --export > requirements.txt
```

to update the environment whenever there's been a change made in the dependencies.

More on conda with GPUS:
 - https://fmorenovr.medium.com/set-up-conda-environment-pytorch-1-7-cuda-11-1-96a8e93014cc

## Sturm-Liouville Problem

Given a differential equation of the form:

$$ -\frac{d}{dx}\left(p(x)\frac{du}{dx}\right) + q(x)u = \lambda w(x)u $$

subject to certain boundary conditions, where:
- $u(x)$ is the unknown function,
- $p(x)$, $q(x)$, and $w(x)$ are given functions,
- $\lambda$ is an eigenvalue parameter.

The boundary conditions are typically specified as:
1. Homogeneous Dirichlet Boundary Conditions: $u(a) = u(b) = 0$
2. Homogeneous Neumann Boundary Conditions: $u'(a) = u'(b) = 0$
3. Mixed Boundary Conditions: $u(a) = 0, \, u'(b) = 0$ or $u'(a) = 0, \, u(b) = 0$

The goal is to find the eigenvalues $\lambda_n$ and corresponding eigenfunctions $u_n(x)$ that satisfy the differential equation and the specified boundary conditions. 

## Instructions 

First, create the dataset using the command
```
python3 dataset/create_dataset.py
```

which will create a tfrecord in a folder named data. This command also takes in three arguments: `number`, `gridsize`, and `path`. The `number` flag represents the number of operators to generate, the `gridsize` flag represents the number of points to be sampled in each operator, and the `path` flag is where the tfrecord dataset will be created. An example command is

```
python3 dataset/create_dataset.py --number 1000 --gridsize 1001 --path ./data
```

To train the model, run the following command in the terminal:
```
python tools/trainer.py
```

and to test the model, 
```
python tools/predictor.py
```

To observe tensorboard logs if enabled, use the following command
```
tensorboard --logdir ./lightning_logs/{current version}
```


## Docker

Run a docker container from a docker image built from the Dockerfile  

```
docker build -t icon .
```

and then run a container using this command

```
docker run --name icon --gpus all -it --rm icon
```

Used these resources to help make dockerfile
 - https://saturncloud.io/blog/how-to-install-pytorch-on-the-gpu-with-docker/
 - https://stackoverflow.com/questions/65492490/how-to-conda-install-cuda-enabled-pytorch-in-a-docker-container

