import os
from typing import Dict
import json
import subprocess
import torch
import pandas as pd
from config import Params
import matplotlib.pyplot as plt
import seaborn as sns


def random_weights(input_size: int = 1,
                   output_size: int = None,
                   sparsity: float = 1,
                   mask: torch.Tensor = None,
                   distribution: str = 'normal',
                   dist_params: [float, float] = [0, 1],
                   spectral_radius: float = None,
                   seed: int = None):
    '''
    Generate random weights according to parameters

    Arguments:
    ----------
    input_size: int - the number of input units
    output_size: int - the number of output units
    sparsity: float in interval [0, 1] - the sparsity of the wright matrix
    mask: torch.tensor of boolean - a mask applied to the weight matrix
    distribution: str among ['normal', 'uniform'] - the type of weight distribution
    dist_params: [mean, std] for normal distrib
                 [min, max] for uniform
    spectral_radius: float in interval [0, inf) - the highest absolute eigen value of the matrix
    seed: float - the seed used to pseudo-randomly generate the weights for reproducibility
    '''
    if input_size is None:
        output_size = input_size

    if spectral_radius is not None and input_size != output_size:
        raise ValueError('spectral_radius can be defined only for square matrices (nbUnitsIN == nbUnitsOUT)')

    if sparsity > 1 or sparsity < 0:
        raise ValueError('sparsity argument is a float between 0 and 1')

    # Set the seed for the random weight generation
    if seed is not None:
        if not isinstance(seed, int):
            raise ValueError('seed must be an integer with base 10')
        else:
            torch.manual_seed(seed)

    # Uniform random distribution of weights:
    if distribution == 'uniform':
        if dist_params is not None:
            minimum, maximum = dist_params
        else:
            minimum, maximum = [-1, 1]
        weights = (torch.rand((output_size, input_size)) * (maximum - minimum) + minimum)

    # Normal (gaussian) random distribution of weights:
    elif distribution == 'normal':
        if dist_params is not None:
            mu, sigma = dist_params
        else:
            mu, sigma = [0, 1]
        weights = torch.randn(output_size, input_size) * sigma + mu

    weights = weights * (torch.rand_like(weights) < sparsity)

    if mask is not None:
        weights = weights * mask

    if spectral_radius is not None:
        currentSpecRad = max(torch.abs(torch.linalg.eig(weights)[0]))
        weights = weights / currentSpecRad * spectral_radius

    return weights


def params_to_dict(params, sequence_item_values, genes):
    return {'optimization_params': params.to_dict(params),
            'sequence_item_values': sequence_item_values,
            'genes': {name: str(gene) for name, gene in genes.items()}}


def write_params(Params, sequence_item_values, genes):
    make_dir()
    params_dict = params_to_dict(Params, sequence_item_values, genes)
    # json_string = json.dumps(params_dict)
    with open(Params.path_config, 'w') as outfile:
        json.dump(params_dict, outfile)


def write_results(results):
    make_dir()
    pd.DataFrame(results).set_index('fitness').to_csv(Params.path_results_csv)
    with open(Params.path_hiplot, 'w') as fout:
        proc = subprocess.Popen(('hiplot-render', Params.path_results_csv), stdout=fout)
        return_code = proc.wait()


def make_dir():
    if not os.path.exists(Params.output_dir):
        os.mkdir(Params.output_dir)


def plot_test(test_results: Dict, sequence_item_values: Dict):
    sns.set(font_scale=1.3, style='whitegrid')
    plt.figure(figsize=(10, 7))
    title = 'Predicted and actual value obtained with best parameters'
    items = list(sequence_item_values.keys())
    items.sort()
    title += '\n' + ' | '.join([f"{item}={sequence_item_values[item]}" for item in items])
    plt.suptitle(title)

    plt.subplot(211)
    target = test_results['target'][:, 0]
    pred = test_results['predictions'][:, 0]
    mse = torch.mean((target-pred)**2)
    plt.title(f"Example sequence 1 | MSE = {mse:.5f}")
    plt.plot(target, label='target')
    plt.plot(pred, label='predictions')
    plt.ylabel('Value of sequence')
    tick_labels = test_results['sequences'][0]
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels)
    plt.legend()

    plt.subplot(212)
    target = test_results['target'][:, 1]
    pred = test_results['predictions'][:, 1]
    mse = torch.mean((target-pred)**2)
    plt.title(f"Example sequence 2 | MSE = {mse:.5f}")
    plt.plot(target)
    plt.plot(pred)
    plt.xlabel('Item in sequence')
    plt.ylabel('Value of sequence')
    tick_labels = test_results['sequences'][1]
    plt.xticks(ticks=range(len(tick_labels)), labels=tick_labels)
    plt.tight_layout()
    plt.show()
