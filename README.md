# Genetic Algorithm

Here is a very simple implementation of a genetic algorithm with an object-oriented easy-to-use interface.

## Basic usage

The most basic usage of the algorithm requires only two objects: `Gene` and `Evolution`.

A `Gene` is a parameter to be optimized whose value is explored within specified
bounds, and can be one of the following type:
- uniform
- uniform integer
- log uniform
- ordinal
- category

Each `Gene` is instantiated with a corresponding static method:

```
alpha = Gene.log_uniform(-5, -1, base=10)
```

All the parameters to be optimized must be compiled into a dictionary of `Gene`s
that is passed to the `Evolution` object.

```
population_size = 100

genes = {'alpha': Gene.log_uniform(-5, -1, base=10),
         'batch_size': Gene.ordinal((8, 16, 32)),
         'dropout': Gene.ordinal((.5, .6, .7, .75, .8, .85, .9)),
         'optimizer': Gene.category(('Adam', 'SGD'))}
         
evo = Evolution(genes, population_size, fitness_function)
```

`population_size` defines how many combinations of parameters are tested at each
iteration of the algorithm with the `fitness_function`. This function takes a
list of parameter combinations and return the fitness for each of them. Ideally
this function test each parameter set in parallel for faster computation.

# Example

This repository contains an optimization example demonstrating how to find
parameters for recurrent neural network associated values to sequences.

## Task

Given a sequence of items that have associated values, the model must learn to
output the cumulative value of the items at each time step. In this example, we
have 5 different items with values varying between -0.2 and 0.2:
```
item_values = {'A': .2, 'B': .1, 'C': 0, 'D': -.1, 'E': -.2}
```

So the sequences `['B', 'E', 'C', 'D']` as a target output of `[.1, -.1, -.1, -.2']`.

To perform this task accurately a recurrent network must integrate all the
previous inputs into a cumulative sum.

## Model

![Schema of an echo state network](https://www.researchgate.net/profile/Joschka-Boedecker/publication/256459964/figure/fig1/AS:298010991972352@1448062765089/The-architecture-of-a-typical-Echo-State-Network-ESN-which-belongs-to-the-class-of.png)

To perform this task we use a [Reservoir recurrent
network](https://en.wikipedia.org/wiki/Reservoir_computing). It has the
particularity that its recurrent weights are not trained, but are randomly
generated according to parameters that endow the network with a given dynamic.
The training is performed on the readout weights by performing a simple
regularized regression between the activity of the recurrent units and the
target output. Here we use an [echo state
network](http://www.scholarpedia.org/article/Echo_state_network) (ESN) version
that implements leaky units (a single unit has a leaky memory of its past
inputs).

The parameters of the network must be optimized depending on the task:
- `nunits`: the number of units in the recurrent layer
- `tau`: the time constant of integration of the leaky units
- `spectral radius`: the highest absolute eigen value of the recurrent matrix
- `ridge parameter`: the regularization parameter used in training of the readout weights

## Files

Here is a list of the source files and what they relate to:
- `leakycell.py` and `leakyRNNs.py` are custom implementations of echo state networks in `PyTorch`
- `genetic_algorithm.py` contains all the classes of the genetic algorithm
- `sequential_task.py` contains all the functions to create inputs and outputs for the sequential task described above
- `config.py` allows you to specify parameters for the example such as the nature of the optimized parameters, and parameters of the optimization itself
- `utils.py` contains helper functions
- `main.py` contains the main script to run the example

## Running the example

### Requirements

The reservoir network has been implemented with `PyTorch`, so this library must
be installed along side others to be able to run the example. All the
requirements are defined in a `.yml` file that has been generated with `conda`.
You can recreate the environment with the following command:
```
conda env create -f requirements.yml
```
This will create a environment named `geneticalgo` that must be activated before running the example:
```
conda activate geneticalgo
```

### Running the example

In the folder `src` the file `main.py` performs the following operations:
- optimize the parameters of the recurrent network
- write the results in `.csv` and interactive `.html` files in an output folder
- plot the results of two examples of the network running with optimized parameters

Simply run `python main.py` in the command line with the proper environment activated.

The `.html` file allows interactive exploration of the optimization results with
the [Hiplot](https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/) tool.

### Customizing the example

In the `config.py` file, parameters can be edited to customize the optimization and the task.
