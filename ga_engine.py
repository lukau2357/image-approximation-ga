import cv2
import numpy as np
import time
import json
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from ellipse_rgb_alpha import PopulationEllipseRGBA
from population import Population
from poly_rgb_alpha import PopulationPolyRBGAlpha

class GAEvolver:
    def __init__(self, alg_label : str, target : np.ndarray, target_path : str, generations : int,
                fittest_survival_ratio: float, population : Population, fitness_label : str, 
                crossover_label : str, caching : bool = False, caching_period : int = 100):

        self.target = target
        self.target_norm = self.target.astype(np.float64) / 255
        self.target_path = target_path
        self.generations = generations
        self.population = population
        self.global_best = None
        self.global_best_f = None
        self.alg_label = alg_label

        self.caching = caching
        self.last_generation_index = 0
        self.caching_period = caching_period

        if fittest_survival_ratio < 0 or fittest_survival_ratio > 1:
            raise Exception("Invalid value for fittes survival ratio given, must be in [0,1].")
        self.fittest_survival_ratio = fittest_survival_ratio

        if fitness_label not in ["psnr", "ssim", "mse"]:
            raise Exception("Invalid fitness label passed.")
        self.fitness_label = fitness_label

        if self.fitness_label == "psnr":
            self.fitness = GAEvolver.fitness_psnr
        
        elif self.fitness_label == "mse":
            self.fitness = GAEvolver.fitness_nmse
        
        else:
            self.fitness = GAEvolver.fitness_ssim
        
        if crossover_label not in ["uniform", "cuts", "id"]:
            raise Exception("Invalid crossover label passed.")
        self.crossover_label = crossover_label

        if crossover_label == "uniform":
            self.crossover = self.population.crossover_uniform
        
        elif crossover_label == "cuts":
            self.crossover = self.population.crossover_cuts
        
        elif crossover_label == "id":
            self.crossover = self.population.crossover_id

        self.f_values = np.empty((self.population.population_size))

    @staticmethod
    def fitness_psnr(target : np.ndarray, image : np.ndarray) -> float:
        return cv2.PSNR(target, image)

    @staticmethod
    def fitness_nmse(target: np.ndarray, image : np.ndarray) -> float:
        return -mse(target, image)

    @staticmethod
    def fitness_ssim(target : np.ndarray, image : np.ndarray) -> float:
        return ssim(target, image, channel_axis = 2)

    def generation_change(self):
        indices = np.argsort(-self.f_values)
        
        # Update the best global configuration
        if self.global_best_f is None or self.global_best_f < self.f_values[indices[0]]:
            self.global_best = self.population.current_generation[indices[0]].copy()
            self.global_best_f = self.f_values[indices[0]]

        # Survival of the fittest
        fittest_threshold = int(self.population.population_size * self.fittest_survival_ratio)
        indices = indices[:fittest_threshold]
        fsum = self.f_values[indices].sum()
        categorical_weights = np.array([item / fsum for item in self.f_values[indices]], dtype = np.float64)
        self.population.current_generation[:fittest_threshold] = self.population.current_generation[indices]
        self.population.drawings[:fittest_threshold] = self.population.drawings[indices]
        self.f_values[:fittest_threshold] = self.f_values[indices]

        # Selection and mating
        size = fittest_threshold
        while size < self.population.population_size:
            parrent_indices = self.population.rand.choice(indices, size = 2, p = categorical_weights, replace = False)
            c1, c2 = self.crossover(self.population.current_generation[parrent_indices[0]], 
                                    self.population.current_generation[parrent_indices[1]],
                                    self.f_values[parrent_indices[0]] / fsum,
                                    self.f_values[parrent_indices[1]] / fsum)

            self.population.current_generation[size] = c1
            size += 1
            if size >= self.population.population_size:
                break

            self.population.current_generation[size] = c2
            size += 1

        # Mutation, every gene of offsprings mutates with small probability
        self.population.mutate_generation(fittest_threshold)

        # Precomputing the corresponding images.
        for i in range(fittest_threshold, self.population.population_size):
            drawing = self.population.embed_chromosome(self.population.current_generation[i])
            self.population.drawings[i] = drawing
            self.f_values[i] = self.fitness(self.target_norm, drawing.astype(np.float64) / 255)

    def evolution(self):
        if self.last_generation_index == 0:
            self.population.generate_initial_population(self.target)
            print("Finished generating initial population.\n")
            for i in range(self.population.population_size):
                self.f_values[i] = self.fitness(self.target_norm, self.population.drawings[i].astype(np.float64) / 255)

        else:
            print("Continuing evolution from cached data.")

        for i in range(self.last_generation_index, self.generations):
            s = time.time()
            self.generation_change()
            e = time.time()

            print("Generation: {:d}".format(i))
            print("Best global fitness: {:.5f}".format(self.global_best_f))
            print("Time taken: {:.2f}\n".format(e - s))

            if self.caching:
                self.export(i)

            self.last_generation_index = i
        
        print("Evolution finished!\n")

    def export(self, generation_index : int):
        if not os.path.exists("./{}".format(self.alg_label)):
            os.mkdir("./{}".format(self.alg_label))
            os.mkdir("./{}/best_chromosomes_history".format(self.alg_label))

        evolver_metadata_path = "./{}/evolver_metadata.json".format(self.alg_label)
        with open(evolver_metadata_path, "w") as f:
            d = {
                "target_path": self.target_path,
                "generations": self.generations,
                "last_generation_index": generation_index,
                "caching": self.caching,
                "caching_period": self.caching_period,
                "fittest_survival_ratio": self.fittest_survival_ratio,
                "fitness_label": self.fitness_label,
                "crossover_label": self.crossover_label,
                "population_label": self.population.__class__.__name__
            }
            json.dump(d, f, indent = 4)

        self.population.export("./{}".format(self.alg_label))

        with open("./{}/f_values.npy".format(self.alg_label), "wb") as f:
            np.save(f, self.f_values)

        if generation_index % self.caching_period == 0 or\
           generation_index == self.generations - 1:
           with \
                open("./{}/best_chromosomes_history/{:d}.npy".format(self.alg_label, generation_index), "wb")\
                as f:
                np.save(f, self.global_best)

    @staticmethod
    def load(root):
        with open("./{}/evolver_metadata.json".format(root), "r") as f:
            data = json.load(f)

        if data["population_label"] == "PopulationPolyRBGAlpha":
            population = PopulationPolyRBGAlpha.load(root)

        elif data["population_label"] == "PopulationEllipseRGBA":
            population = PopulationEllipseRGBA.load(root)
        
        target = cv2.imread(data["target_path"])
        res = GAEvolver(root, target, data["target_path"], data["generations"], 
                       data["fittest_survival_ratio"], population, data["fitness_label"], 
                       data["crossover_label"], caching = True, caching_period = data["caching_period"])

        res.last_generation_index = data["last_generation_index"] + 1
        print("Finished loading the cached GAEvolver object.")

        with open("./{}/f_values.npy".format(root), "rb") as f:
            res.f_values = np.load(f)

        print("Finished loading the cached f-values.\n")
        return res

    def render(self, generation_index):
        with open("./{}/best_chromosomes_history/{:d}.npy".format(self.alg_label, generation_index), "rb") as f:
            chromosome = np.load(f)
        
        canvas = self.population.embed_chromosome(chromosome).astype(np.uint8)
        cv2.imshow("./{}-{:d}".format(self.alg_label, generation_index), canvas)
        cv2.waitKey(0)