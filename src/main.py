import numpy as np
from genetic_algorithm import Evolution
from sequential_task import generate_train_test, reservoir_fitness, test_reservoir
from config import Params, sequence_item_values, genes
from utils import write_params, write_results, plot_test


def test_fitness_function(param_list):
    '''Test list of parameters with fitness function

    Parameters
    ----------
    param_list : List
    '''
    results = []
    for _ in range(Params.nrepetitions):
        results.append([reservoir_fitness(data, **params) for params in param_list])
    scores = [np.mean(ind) for ind in zip(*results)]
    return scores


if __name__ == "__main__":
    # Generate train/test data of sequences and their corresponding values
    data = generate_train_test(Params.train_size, Params.test_size, sequence_item_values)

    # Initialize and start the genetic algorithm
    evo = Evolution(genes, Params.population_size, test_fitness_function)
    evo.start(Params.ngenerations, Params.threshold)

    # Gather all results including parameters and associated fitness
    all_results = evo.get_all_params()
    all_results = {param: [ind[param] for ind in all_results] for param in all_results[0].keys()}

    # Write the parameters and fitness into a .csv file and generate a html file to visualize the results
    write_params(Params, sequence_item_values, genes)
    write_results(all_results)

    # Test the network on best parameters and plot the results into a file
    best_params = evo.get_best().get_parameters()
    test_results = test_reservoir(Params.train_size, sequence_item_values, **best_params)
    plot_test(test_results, sequence_item_values)
