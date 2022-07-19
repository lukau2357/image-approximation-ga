import numpy as np

from abc import ABCMeta, abstractmethod

class Population(metaclass = ABCMeta):
    def __init__(self, width : int, height : int, population_size : int, 
                 expected_number_of_mutations : int, rand_seed : int):
        self.width = width
        self.height = height
        self.population_size = population_size
        self.expected_number_of_mutations = expected_number_of_mutations
        self.rand_seed = rand_seed
        self.rand = np.random.default_rng(rand_seed)
        self.feature_space_dim = None
        self.current_generation = None
        self.drawings = None
        
    @abstractmethod
    def generate_initial_population(self, target : np.ndarray = None):
        pass
    
    @abstractmethod
    def embed_chromosome(self, chromosome : np.ndarray):
        pass

    @abstractmethod
    def crossover_cuts(self, *args):
        pass

    @abstractmethod
    def crossover_uniform(self, *args):
        pass

    # Identity crossover
    def crossover_id(self, *args):
        return args[0].copy(), args[1].copy()

    @abstractmethod
    def mutate(self, chromosome : np.ndarray):
        pass
    
    @abstractmethod
    def mutate_generation(self, mutation_threshold):
        pass

    @abstractmethod
    def export(self, path):
        pass