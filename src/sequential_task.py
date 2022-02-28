from typing import Dict, List, Tuple, NewType
from dataclasses import dataclass
from random import choice
import torch
from leakyRNNs import Reservoir


WEIGHT_SEED = 32949857
SEQ_LENGTH = 15

TorchTensor = NewType('torch_tensor', torch.Tensor)


@dataclass
class Data:
    input_train: TorchTensor
    input_test: TorchTensor
    target_train: TorchTensor
    target_test: TorchTensor


def generate_seq_and_out(length: int, sequence_item_values: Dict[str, float]) -> Tuple[List, List]:
    '''Generate sequences and corresponding value output

    Parameters
    ----------
    length : int
      The length of the sequence to be generated
    sequence_item_values : dict[str, float]
      Keys are the possible item names (e.g. 'A', 'B', etc.) and values are the
      'value' of each item

    Returns
    -------
    sequence : List
      A list of strings corresponding to each item of the generated sequence
    values : List
      List a list of cumulative values
    '''
    value_current = 0
    sequence = []
    values = []
    items = list(sequence_item_values.keys())
    for _ in range(length):
        item = choice(items)
        sequence.append(item)
        value_current += sequence_item_values[item]
        values.append(value_current)
    return sequence, values


def seq_to_tensor(seq: List[str]) -> TorchTensor:
    '''Transform a sequence of items into a torch tensor input'''
    A = torch.zeros((1, 5))
    B = torch.zeros((1, 5))
    C = torch.zeros((1, 5))
    D = torch.zeros((1, 5))
    E = torch.zeros((1, 5))

    A[0, 0] = 1
    B[0, 1] = 1
    C[0, 2] = 1
    D[0, 3] = 1
    E[0, 4] = 1

    ELT_TO_VECTOR = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

    return torch.cat([ELT_TO_VECTOR[elt] for elt in seq], 0)


def generate_input_target(size, sequence_item_values: Dict[str, float]) -> Tuple[TorchTensor, TorchTensor]:
    '''Generate input and target tensors for training or testing

    Parameters
    ----------
    size : int
      The number of sequences to generate
    sequence_item_values : dict[str, float]
      Keys are the possible item names (e.g. 'A', 'B', etc.) and values are the
      'value' of each item

    Returns
    -------
    input_ : torch.Tensor
      An torch tensor input to the network
    target : torch.Tensor
      An torch tensor target to train the network
    '''
    seq, values = zip(*[generate_seq_and_out(SEQ_LENGTH, sequence_item_values) for _ in range(size)])
    input_ = torch.cat([seq_to_tensor(seq).unsqueeze(1) for seq in seq], 1)
    target = torch.cat([torch.tensor(values)[:, None, None] for values in values], 1)
    return input_, target


def generate_train_test(train_size: int, test_size: int, sequence_item_values: Dict[str, float]) -> Data:
    '''Generate training and testing data'''
    input_train, target_train = generate_input_target(train_size, sequence_item_values)
    input_test, target_test = generate_input_target(test_size, sequence_item_values)

    return Data(input_train=input_train,
                input_test=input_test,
                target_train=target_train,
                target_test=target_test)


def train_reservoir(input_: int, target: int,
                    reservoir_size: int = None,
                    tau: float = None,
                    spectral_radius: float = None,
                    ridge_param: float = None) -> Reservoir:
    '''Train a reservoir and returns it'''
    reservoir = Reservoir(input_size=5, hidden_size=reservoir_size, tau=tau,
                          weight_seed=WEIGHT_SEED,
                          spectral_radius=spectral_radius,
                          actfout=torch.tanh)

    reservoir.train(input_, target, ridge=ridge_param)
    return reservoir


def reservoir_fitness(data: Data, **parameters) -> float:
    '''Get the fitness of a reservoir network with a given set of parameters'''
    reservoir = train_reservoir(data.input_train, data.target_train, **parameters)
    predictions = reservoir(data.input_test)[0]

    target_test = data.target_test

    # Fitness is the negative MSE and a penalty on the number of units in the reservoir
    mse = ((predictions - target_test)**2).mean()
    fitness = - mse - parameters['reservoir_size']/10**5

    return fitness


def test_reservoir(train_size: int, item_values: Dict, **params) -> Dict:
    '''Testing a reservoir network on two sequences with a given set of parameters'''
    input_train, target_train = generate_input_target(train_size, item_values)
    sequences, values = zip(*[generate_seq_and_out(SEQ_LENGTH, item_values) for _ in range(2)])
    input_test = torch.cat([seq_to_tensor(seq).unsqueeze(1) for seq in sequences], 1)
    target_test = torch.cat([torch.tensor(values)[:, None, None] for values in values], 1)

    reservoir = train_reservoir(input_train, target_train, **params)
    predictions = reservoir(input_test)[0]
    return {'sequences': sequences,
            'predictions': predictions.squeeze(),
            'target': target_test.squeeze()}
