from typing import Dict
from pathlib import Path
from dataclasses import dataclass
from genetic_algorithm import Gene
from datetime import datetime


_working_dir = Path(__file__).parent.parent
_time_stamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")


@dataclass()
class Params:
    '''Parameters of a genetic algorithm evolution'''
    population_size: int = 50  # minimum 10

    # The number of sequences used to train and test the network
    train_size: int = 100
    test_size: int = 50

    # The maximum number of generations/iterations
    ngenerations: int = 100

    # A fitness threshold for early stopping of the evolution: the average
    # difference between the best fitness for 5 successive generations
    threshold: int = .0001

    # The number of times a set of parameters is tested, fitness is the average of the repeated tests
    nrepetitions: int = 5

    output_dir: Path = Path(_working_dir / "output")
    path_results_csv: Path = Path(_working_dir / "output" / ("reservoir_param_search_" + _time_stamp + ".csv"))
    path_hiplot: Path = Path(_working_dir / "output" / ("reservoir_param_search" + _time_stamp + ".html"))
    path_config: Path = Path(_working_dir / "output" / ("optimization_params" + _time_stamp + ".json"))

    def to_dict(self) -> Dict:
        return {'population_size': self.population_size,
                'train_size': self.train_size,
                'test_size': self.test_size,
                'ngenerations': self.ngenerations,
                'threshold': self.threshold,
                'nrepetitions': self.nrepetitions}


# The value of each item in the sequence
sequence_item_values = {'A': .2,
                        'B': .1,
                        'C': .0,
                        'D': -.1,
                        'E': -.2}

# Here define the boundaries and type of parameters to optimize
genes = {'reservoir_size':  Gene.uniform_integer(1, 500),
         'tau':             Gene.uniform(1, 10),
         'spectral_radius': Gene.uniform(.1, 2.),
         'ridge_param':     Gene.log_uniform(-7, 7, base=10)}
