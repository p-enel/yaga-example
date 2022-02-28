from typing import Callable, Union, Sequence, Dict, List, Tuple, Any
import numpy as np


class Gene:
    """A Gene correspond to one parameter to be optimized

    There are 5 types of available genes with the current implementation, it
    can easily be extended. Below are the available Genes followed by their
    arguments:

    - uniform : a uniform distribution (float)
      low : float - the lower boundary
      high : float - the higher boundary
    - uniform_integer : a discrete uniform set of integers
      low : int - the lower boundary
      high : int - the higher boundary
    - log_uniform : a log uniform distribution (float)
      low : int - the lower boundary
      high : int - the higher boundary
      base [optional, default=10] : float - the base of the logarithm
    - ordinal : a discrete ordered set of specified integers or strings
      values : list/tuple/range of int or strings - an ordered
        sequence of values
    - category : a set of unordered values of any type
      categories : list/tuple of int or strings - a set of values

    To create a new Gene, use one of the above static method with the
    appropriate arguments.

    e.g., Gene.uniform(-1, 1) for a gene with a uniform distribution
    between -1 and 1.
    """
    def __init__(self, type_):
        self.type = type_
        self.p = None
        self.generate = None
        self.repr = None

    @staticmethod
    def uniform(low: float, high: float) -> 'Gene':
        """Uniform real value Gene

        Parameters
        ----------
        low : float
          The lower bound including this value
        high : float
          The upper bound including this value

        Returns
        -------
        Gene
          A uniform Gene
        """
        gene = Gene('uniform')
        gene.p = [low, high]
        gene.generate = gene._uniform_gen
        gene.repr = '%.2f'
        return gene

    def _uniform_gen(self, value: float = None, within: float = 1) -> float:
        """Sample a real value from a uniform distribution

        Parameters
        ----------
        value : float, optional
          Any real value within the bounds defined with the Gene. If no value
          is specified, a value will be sampled from the bounded uniform
          distribution.
        within : float [0 -> 1], default=1
          If value is specified, the sample is 'within' distance of value,
          where 0 means that 'value' will be chosen and 1 means that any value
          within the full range of values centered on 'value'. I.e. if 'value'
          is at the end of the range, only half of the values are considered.
        """
        if value is None:
            return np.random.uniform(low=self.p[0], high=self.p[1])

        assert self.p[0] < value < self.p[1]
        assert 0 <= within <= 1
        new_range = (self.p[1] - self.p[0]) * within
        low = value - new_range / 2
        high = value + new_range / 2
        low = self.p[0] if low < self.p[0] else low
        high = self.p[1] if high > self.p[1] else high
        return np.random.uniform(low=low, high=high)

    @staticmethod
    def uniform_integer(low: int, high: int) -> 'Gene':
        """Uniform integer value Gene

        Parameters
        ----------
        low : int
          The lower bound including this value
        high : int
          The upper bound including this value

        Returns
        -------
        Gene
          A uniform integer Gene
        """
        gene = Gene('uniform integer')
        gene.p = [low, high]
        gene.generate = gene._uniform_integer_gen
        gene.repr = '%d'
        return gene

    def _uniform_integer_gen(self, value: int = None, within: float = 1) -> int:
        """Sample an integer from a uniform discrete distribution

        Parameters
        ----------
        value : int, optional
          Any integer within the bounds defined with the Gene. If no value is
          specified an integer will be sampled from the bounded distribution.
        within : float [0 -> 1], default=1
          If value is specified, the sample is 'within' distance of value,
          where 0 means that 'value' will be chosen and 1 means that any value
          within the full range of values centered on 'value'. I.e. if 'value'
          is at the end of the range, only half of the values are considered.
        """
        if value is None:
            return np.random.randint(self.p[0], high=self.p[1]+1)

        assert self.p[0] <= value <= self.p[1]
        assert 0 <= within <= 1
        new_range = (self.p[1] - self.p[0]) * within
        low = int(np.round(value - new_range / 2))
        high = int(np.round(value + new_range / 2))
        low = self.p[0] if low < self.p[0] else low
        high = self.p[1] if high > self.p[1] else high
        return np.random.randint(low, high=high+1)

    @staticmethod
    def log_uniform(low: float, high: float, base: float = 10) -> 'Gene':
        """Log uniform real values

        Pick a value from a log uniform distribution

        Parameters
        ----------
        low : float
          The lower bound of the power of base 'base'
        high : float
          The upper bound of the power of base 'base'
        base : float, default=10
          The base of the power used for log uniform values. Default is base 10

        Returns
        -------
        Gene
          A log uniform Gene
        """
        gene = Gene('log uniform')
        gene.p = [low, high, float(base)]
        gene.generate = gene._log_uniform_gen
        gene.repr = '%.2e'
        return gene

    def _log_uniform_gen(self, value: float = None, within: float = 1) -> float:
        """Sample a real value from a log uniform distribution

        Parameters
        ----------
        value : float, optional
          Any value within the bounds defined with the Gene. If 'value' is not
          specified, any value within the bounded distribution will be sampled.
        within : float [0 -> 1], default=1
          If value is specified, the sample is 'within' distance of value,
          where 0 means that 'value' will be chosen and 1 means that any value
          within the full range of values centered on 'value'. I.e. if 'value'
          is at the end of the range, only half of the values are considered.
        """
        if value is None:
            return self.p[2]**np.random.uniform(self.p[0], self.p[1])

        exponent = np.log(value)/np.log(self.p[2])
        assert self.p[0] <= exponent <= self.p[1]
        power = self._uniform_gen(value=exponent, within=within)
        return self.p[2]**power

    @staticmethod
    def ordinal(values: Sequence[Union[str, int]]) -> 'Gene':
        """Ordinal with integer values

        Parameters
        ----------
        values : Sequence[Union[str, int]]
          A tuple or list of str's or integers

        Returns
        -------
        Gene
          An ordinal Gene
        """
        gene = Gene('ordinal')
        gene.p = tuple(values)
        try:
            assert np.all([isinstance(val, type(values[0])) for val in values[1:]])
            assert type(values[0]) in [int, str]
        except AssertionError:
            ValueError('Ordinal gene takes a sequence of str or int as argument')

        gene.generate = gene._ordinal_gen
        gene.repr = '%d' if isinstance(values[0], int) else '%s'
        return gene

    def _ordinal_gen(self, value: [Union[str, int]] = None, within: float = 1):
        """Sample an ordinal

        Parameters
        ----------
        value : [Union[str, int]], optional
          One of the possible values that this Gene can take (as defined when
          creating the gene). If no 'value' is specified, one of the possible
          values will be picked with equal probability.
        within : float [0 -> 1], default=1
          If value is specified, the sample is 'within' distance of value,
          where 0 means that 'value' will be chosen and 1 means that any value
          within the full range of values centered on 'value'. I.e. if 'value'
          is at the end of the range, only half of the values are considered.
        """
        if value is None:
            return np.random.choice(self.p)

        assert value in self.p
        assert 0 <= within <= 1
        if isinstance(within, float):
            within = int(np.round(within*len(self.p)))
        index = self.p.index(value)
        ilow = index - within
        ihigh = index + within + 1
        ilow = ilow if ilow >= 0 else 0
        ihigh = ihigh if ihigh <= len(self.p) else len(self.p)
        return np.random.choice(self.p[ilow:ihigh])

    @staticmethod
    def category(categories: Sequence[str]) -> 'Gene':
        """Categorical gene that can take any type of value

        Parameters
        ----------
        categories : Sequence[str]
          A sequence of values of any type

        Returns
        -------
        Gene
          A category Gene
        """
        gene = Gene('category')
        gene.p = tuple(categories)
        gene.generate = gene._category_gen
        gene.repr = '%s'
        return gene

    def _category_gen(self, value: Any = None, within: float = 1):
        '''Sample a category

        Parameters
        ----------
        value : Any, optional
          Any value corresponding to a category. If no 'value' is specified,
          one of the possible values will be picked with equal probability.
        within : float, default=1
          Parameter related to the probability of choosing 'value':
          0 means 'value' is chosen with probability 1, 1 means 'value' is
          chosen with equal probability with respect to the other categories.
          Any number between 0 and 1 is a compromise between these two cases.
        '''
        if value is None:
            return np.random.choice(self.p)

        assert value in self.p
        assert 0 <= within <= 1
        p = self.p.copy()
        p.remove(value)
        ncats = len(self.p)
        if np.random.rand() < (1-ncats)/ncats * within + 1:
            return value
        else:
            return np.random.choice(p)

    def __repr__(self) -> str:
        return f"Gene <{self.type}>, params {self.p}"


class Genome:
    def __init__(self, genes: Dict[str, Gene], generate: bool = True):
        """An ensemble of Genes/parameters that can be tested for fitness

        Parameters
        ----------
        genes : Dict[str, Gene]
          A dictionary with a names as keys and Genes as values
        generate : bool, default=True
          Whether to automatically generate a value for each gene
        """
        assert isinstance(genes, dict)
        self.genes = genes
        self.fitness = None
        if generate:
            self.data = self.generate()

    def generate(self) -> Dict[str, Union[str, float, int]]:
        """Instantiate each gene value according to gene templates

        Returns
        -------
        Dict[str, Union[str, float, int]]
          A dictionary containing the value for each gene
        """
        new_genome = {name: gene.generate() for name, gene in self.genes.items()}
        return new_genome

    def mutate(self, mutations: Dict[str, float] = {}):
        """Mutate genes by a given range

        Parameters
        ----------
        mutations : Dict[str, float], optional
          Dictionary specifying the genes and range [0 -> 1] of the mutation
          for each of them
        """
        for gene, mut in mutations.items():
            self.data[gene] = self.genes[gene].generate(self.data[gene], mut)

    def __repr__(self) -> str:
        genes_str = ''
        for gene, value in self.data.items():
            genes_str += f'{gene} = {value}\n'
        if self.fitness is not None:
            genes_str += f'Fitness -> {self.fitness}'
        return f"Genome with genes:\n{genes_str}"

    def copy(self) -> 'Genome':
        """Make a clean copy of a Genome"""
        new_geno = Genome(self.genes, generate=False)
        new_geno.data = self.data
        new_geno.fitness = self.fitness
        return new_geno

    @staticmethod
    def child(genome1: 'Genome', genome2: 'Genome') -> 'Genome':
        """Get a full crossover of two individuals/genomes

        Parameters
        ----------
        genome1 : Genome
        genome2 : Genome

        Returns
        -------
        Genome
          A new Genome recombined from the two arguments
        """
        child = Genome(genome1.genes, generate=False)
        child.data = {}
        genomes = [genome1, genome2]
        for gene in genome1.genes.keys():
            np.random.shuffle(genomes)
            child.data[gene] = genomes[0].data[gene]
        return child

    def _set_fitness(self, fitness: float) -> None:
        self.fitness = fitness

    def delete_fitness(self) -> None:
        """Delete the fitness, replacing it by None"""
        self.fitness = None

    def get_parameters(self) -> Dict[str, Union[str, float, int]]:
        """Returns the parameters of the genome in a dictionary

        Returns
        -------
        Dict[str, Union[str, float, int]]
          A dictionary of parameters
        """
        return self.data


class Population:
    def __init__(self, genes: Dict[str, Gene], size: int):
        """A set of Genomes or parameters that can be tested and evolved

        Parameters
        ----------
        genes : Dict[str, Gene]
          A dictionary of the genes used to spawn a population
        size : int
          The number of individuals in the population
        """
        self.genes = genes
        self.size = size
        self.generation = 0
        self.assessed = False
        self.individuals = None
        self._fitness_fct = None

    def instantiate(self) -> None:
        """Instantiate a set of individuals based on the gene templates"""
        self.individuals = [Genome(self.genes) for i in range(self.size)]

    def get_individuals(self) -> List[Dict[str, Union[str, int, float]]]:
        """Returns the individuals in a list

        Returns
        -------
        List[Dict[str, Union[str, int, float]]]
        """
        return [self.individuals[ind].data for ind in range(self.size)]

    def mutate(self, ind: int, gene: str, mutation: float) -> None:
        """Mutate one individual on one gene

        Parameters
        ----------
        ind : int
          The index of the individual in the population
        gene : str
          The name of the gene
        mutation : float [0 -> 1]
          A number between 0 and 1 included that specify the range of the mutation,
          0 means no mutation, 1 means a mutation of the widest range possible
          around the current value
        """
        self.individuals[ind].mutate({gene: mutation})

    def mutate_whole_pop(self, pop_portion: float, pct_genes: float, pct_mut: float) -> None:
        """Mutate the whole population according to statistics

        Parameters
        ----------
        pop_portion : float
          The fraction of the population that will undergo mutations
        pct_genes : float
          The fraction of genes that will undergo mutations
        pct_mut : float [0 -> 1]
          The range of the mutation, see method .mutate for more details
        """
        inds = np.arange(self.size)
        np.random.shuffle(inds)
        ninds = int(np.round(pop_portion * self.size))
        ngenes = int(np.round(pct_genes * len(self.genes)))
        genes_n = list(self.genes.keys())
        for ind in inds[:ninds]:
            np.random.shuffle(genes_n)
            for gene in genes_n[:ngenes]:
                self.mutate(ind, gene, pct_mut)
        self.delete_fitness()

    def _increment_generation(self) -> None:
        self.generation += 1

    def get_top(self, top_pct: float) -> List[Genome]:
        """Get the best performing individuals of the population

        Parameters
        ----------
        top_pct : float
          The fraction of top individuals to return

        Returns
        -------
        List[Genome]
          A list of the top individuals
        """
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        ntop = int(np.round(top_pct * self.size))
        return sorted_pop[:ntop]

    def _children(self, top_pop: List[Genome]) -> List[Genome]:
        """Create children by recombining top individuals

        Parameters
        ----------
        top_pop : List[Genome]
          A list of the top individuals

        Returns
        -------
        List[Genome]
          A list of new individuals recombined from top_pop. The size of this
          list is equal to the size of the population minus the size of top_pop
        """
        nchildren = self.size - len(top_pop)
        top_inds = np.arange(len(top_pop))
        children = []
        for ichild in range(nchildren):
            np.random.shuffle(top_inds)
            children.append(Genome.child(top_pop[top_inds[0]], top_pop[top_inds[1]]))
        return children

    def reproduce(self, top_pct: float) -> None:
        """Keep the top individuals and regenerate the population by recombining them

        Parameters
        ----------
        top_pct : float
          The fraction of top individuals to use
        """
        top_pop = self.get_top(top_pct)
        children = self._children(top_pop)
        self.individuals = top_pop + children
        self.delete_fitness()

    def new_generation(self, top_pct: float = .2, pct_pop_mut: float = .3,
                       pct_genes_mut: float = .5, pct_mut: float = .2) -> 'Population':
        """Generates a new generation by recombination of top individuals and mutations

        Parameters
        ----------
        top_pct : float [0 -> 1], default=.2
          The fraction of top individuals for recombination
        pct_pop_mut : float [0 -> 1], default=.3
          The fraction of the population to mutate after recombination
        pct_genes_mut : float [0 -> 1], default=.5
          The fraction of genes to mutate
        pct_mut : float [0 -> 1], default=.2
          The range of the mutation, see method mutate for more details

        Returns
        -------
        Population
          A new recombined and mutated population
        """
        new_pop = self.copy()
        new_pop._increment_generation()
        new_pop.reproduce(top_pct)
        new_pop.mutate_whole_pop(pct_pop_mut, pct_genes_mut, pct_mut)
        return new_pop

    def copy(self) -> 'Population':
        """Make a clean copy of a population"""
        new_pop = Population(self.genes, self.size)
        new_pop.generation = self.generation
        if self.individuals is not None:
            new_pop.individuals = [ind.copy() for ind in self.individuals]
        new_pop._fitness_fct = self._fitness_fct
        new_pop.assessed = self.assessed
        return new_pop

    def set_fitness_fct(self, fitness_fct: Callable[[Dict[str, Union[str, int, float]]], float]) -> None:
        """The fitness function must take a list of dictionaries as argument"""
        self._fitness_fct = fitness_fct

    def assess_pop(self) -> None:
        """Get the fitness for each individual in the population"""
        parameters = [ind.data for ind in self.individuals]
        fitness = self._fitness_fct(parameters)
        assert len(fitness) == len(parameters)
        for ind, fit in zip(self.individuals, fitness):
            fit = fit if fit != np.nan else -np.infty
            ind._set_fitness(fit)
        self.assessed = True

    def delete_fitness(self) -> None:
        """Delete the fitness of each individual"""
        for ind in self.individuals:
            ind.delete_fitness()
        self.assessed = False

    def get_fitness(self) -> List[float]:
        """Get the fitness of each individual

        Returns
        -------
        List[float]
        """
        return [ind.fitness for ind in self.individuals]

    def get_best(self) -> Genome:
        """Returns the best individual of the population

        Returns
        -------
        Genome
        """
        assert self.assessed
        bestid = np.argmax([ind.fitness for ind in self.individuals])
        return self.individuals[bestid]

    def __repr__(self) -> str:
        repr = f"Population of {self.size} individuals, generation {self.generation}"
        repr += f" with genes: {list(self.genes.keys())}"
        if self._fitness_fct is None:
            repr += "\nNo fitness function has been set up"
        else:
            repr += f"\nFitness function: {self._fitness_fct}"
        if self.assessed:
            repr += f"\nBest fitness is {np.max(self.get_fitness())}"
        else:
            repr += "\nNot tested yet"

        return repr

    def __getitem__(self, arg: int) -> Union[Genome, List[Genome]]:
        return self.individuals[arg]

    def get_pop_params(self) -> List[Dict[str, Union[str, float, int]]]:
        """Get the parameters of all individuals in the population

        Returns
        -------
        List[Dict[str, Union[str, float, int]]]
          A list of dictionaries of parameters. Each element of the list is a
          different individual of the population
        """
        assert self.individuals is not None
        pop_params = []
        for ind in self.individuals:
            params = ind.get_parameters().copy()
            params['fitness'] = ind.fitness
            pop_params.append(params)
        return pop_params


class Evolution:
    def __init__(self,
                 genes: Dict[str, Gene],
                 popsize: int,
                 fitness_fct: Callable[[Dict[str, Union[str, int, float]]], float]):
        """Object running the genetic algorithm

        Parameters
        ----------
        genes : Dict[str, Gene]
          A dictionary of the genes used to spawn a population
        popsize : int
          The number of individuals in the population
        fitness_fct : Callable[[Dict[str, Union[str, int, float]]], float]
          The objective function taking a dictionary as argument and returning a float.
          Note that the function is to be maximized NOT minimized.
        """
        self.genes = genes
        assert popsize >= 10
        self.popsize = popsize
        self._fitness_fct = fitness_fct
        self.pop_hist = []
        self.fit_hist = []
        self.stop_cond = None

    def _init_pop(self, genes: Dict[str, Gene], popsize: int):
        self.population = Population(genes, popsize)
        self.population.instantiate()
        self.population.set_fitness_fct(self._fitness_fct)

    @staticmethod
    def _process_niter_threshold(niter: int = None, threshold: float = None) -> Tuple[int, float]:
        if niter is None and threshold is None:
            ValueError("One of 'niter' or 'threshold' argument must be set")
        if niter is None:
            niter = np.infty
        if threshold is None:
            threshold = np.infty
        return niter, threshold

    @staticmethod
    def _start_msg(niter: int, threshold: float, state: str = 'start') -> None:
        state_str = 'Starting' if state == 'start' else 'Resuming'
        start_msg = f"{state_str} evolution with"
        if niter is not np.infty:
            start_msg += f" {niter} generations"
        if threshold is not np.infty:
            start_msg += f" threshold {threshold}"
        print(start_msg)

    def _end_msg(self) -> None:
        msg = f"\n{''.join(['=']*80)}\n"
        msg += (f"Evolution finished after {self.population.generation} iterations"
                f" and with an average fitness change of {self._running_fit()}\n"
                f"The best fitness was {np.max(self.get_best().fitness)} obtained"
                f" by individual {self.get_best()}")
        print(msg)

    def start(self, niter: int = None, threshold: float = None) -> None:
        """Launch the genetic algorithm

        Parameters
        ----------
        niter : int
          The number of generations if threshold is not set, otherwise, the maximum
          number of generations if threshold is set
        threshold : float
          The maximum change in fitness averaged over 5 consecutive generations
        """
        niter, threshold = self._process_niter_threshold(niter, threshold)
        self._init_pop(self.genes, self.popsize)
        self._start_msg(niter, threshold)
        self.stop_cond = self._evo_loop(niter, threshold)
        self._end_msg()

    def _evo_loop(self, niter: int, threshold: float) -> str:
        fit_change = threshold + 1
        while self.population.generation < niter and fit_change > threshold:
            self.population.assess_pop()
            best_fit = np.max(self.population.get_fitness())
            self.fit_hist.append(best_fit)
            fit_change = self._running_fit()
            self._print_evo()
            self._evolve_popluation(niter)
        return 'iter' if self.population.generation < niter else 'threshold'

    def _running_fit(self) -> float:
        if len(self.fit_hist) < 5:
            return np.infty
        else:
            return np.sum(np.abs(np.diff(self.fit_hist[-5:])))

    def _evolve_popluation(self, niter: int) -> None:
        if self.population.generation == 0:
            print('\nUsing the default evolution algorithm. If you want to '
                  'specify your own, inherit Evolution and modify the '
                  'evolve_population method')
        if niter is not np.infty:
            pct_mut = (1 - self.population.generation / niter) / 3
        else:
            pct_mut = .15
        new_pop = self.population.new_generation(top_pct=.2, pct_pop_mut=.5, pct_genes_mut=.2, pct_mut=pct_mut)
        self.pop_hist.append(self.population)
        self.population = new_pop

    def _print_evo(self) -> None:
        msg = f"{''.join(['=']*80)}\n"
        msg += (f"Generation {self.population.generation} -> "
               f"fitness = {self.fit_hist[-1]}")
        msg += (f"\nBest parameters: {self.population.get_best()}")
        print(msg)

    def resume(self, niter: int, threshold: float) -> None:
        """Resume after initial evolution

        Parameters
        ----------
        niter : int
          The number of generations if threshold is not set, otherwise, the maximum
          number of generations if threshold is set
        threshold : float
          The maximum change in fitness averaged over 5 consecutive generations
        """
        if len(self.pop_hist) == 0:
            print("There is no initial evolution, use the method .start(niter, threshold) instead")
            return
        niter, threshold = self._process_niter_threshold(niter, threshold)
        if threshold > self._running_fit():
            print("The specified threshold is higher than the current difference in fitness")
            return
        self._start_msg(niter, threshold, state='resume')
        self.stop_cond = self._evo_loop(niter, threshold)

    def get_best(self) -> Genome:
        """Returns the best performing Genome after evolution

        Returns
        -------
        Genome
        """
        best_pop_id = np.argmax(self.fit_hist)
        best_pop = self.pop_hist[best_pop_id]
        best_ind_id = np.argmax(best_pop.get_fitness())
        return best_pop[best_ind_id]

    def get_all_params(self) -> List[List[Dict[str, Union[str, float, int]]]]:
        """Returns all the parameters of all individuals from all populations

        Returns
        -------
        List[List[Dict[str, Union[str, float, int]]]]
          A list of list of dictionaries of parameters
          The outer list contains successive populations, the inner list
          contains each individual of that population.
        """
        all_params = []
        for pop in self.pop_hist:
            all_params += pop.get_pop_params()
        return all_params

    def __repr__(self) -> None:
        pass
