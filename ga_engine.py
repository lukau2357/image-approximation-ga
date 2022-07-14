import cv2
import numpy as np
import time
import json
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from population import Population
from population_poly import PopulationPolyRBG
from utils import kmeans_color_palette

class GAEvolver:
    def __init__(self, alg_label : str, target : np.ndarray, target_path : str, generations : int,
                fittest_survival_ratio: float, population : Population, 
                fitness_label : str, crossover_label : str, kmeans_init : bool, 
                kmeans_clusters : int = 10, kmeans_iterations : int = 100, 
                kmeans_repeats : int = 10, caching : bool = False, caching_ratio_period : float = 0.1):

        self.target = target
        self.target_path = target_path
        self.generations = generations
        self.population = population
        self.global_best = None
        self.global_best_f = None
        self.alg_label = alg_label

        self.kmeans_init = kmeans_init
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_iterations = kmeans_iterations
        self.kmeans_repeats = kmeans_repeats
        self.caching = caching
        self.last_generation_index = 0

        if caching_ratio_period < 0 or caching_ratio_period > 1:
            raise Exception("Invalid value for caching ratio period given, must be in [0,1].")
        self.caching_ratio_period = caching_ratio_period

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
        
        if crossover_label not in ["uniform", "cuts"]:
            raise Exception("Invalid crossover label passed.")
        self.crossover_label = crossover_label

        if crossover_label == "uniform":
            self.crossover = self.population.crossover_uniform
        
        else:
            self.crossover = self.population.crossover_cuts

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
        target_norm = self.target.astype(np.float64) / 255.0
        f_values = np.empty(self.population.population_size, dtype = np.float64)

        # Fitness calculation
        for i in range(self.population.population_size):
            f_values[i] = self.fitness(target_norm, 
                                       self.population.drawings[i].astype(np.float64) / 255.0)

        indices = np.argsort(-f_values)
        
        # Update the best global configuration
        if self.global_best_f is None or self.global_best_f < f_values[indices[0]]:
            self.global_best = self.population.current_generation[indices[0]].copy()
            self.global_best_f = f_values[indices[0]]

        # Survival of the fittest
        fittest_threshold = int(self.population.population_size * self.fittest_survival_ratio)
        indices = indices[:fittest_threshold]
        fsum = f_values[indices].sum()
        categorical_weights = np.array([item / fsum for item in f_values[indices]], dtype = np.float64)
        self.population.current_generation[:fittest_threshold] = self.population.current_generation[indices]
        self.population.drawings[:fittest_threshold] = self.population.drawings[indices]

        # Selection and mating
        size = fittest_threshold
        while size < self.population.population_size:
            parrent_indices = self.population.rand.choice(indices, size = 2, p = categorical_weights)
            c1, c2 = self.crossover(self.population.current_generation[parrent_indices[0]], 
                                    self.population.current_generation[parrent_indices[1]],
                                    f_values[parrent_indices[0]] / fsum,
                                    f_values[parrent_indices[1]] / fsum)

            self.population.current_generation[size] = c1
            self.population.current_generation[size + 1] = c2
            size += 2
        
        # Mutation, every gene of offsprings mutates with small probability
        for i in range(fittest_threshold, self.population.population_size):
            self.population.mutate(self.population.current_generation[i])
            drawing = self.population.embed_chromosome(self.population.current_generation[i])
            self.population.drawings[i] = drawing

    def evolution(self):
        if self.last_generation_index == 0:
            if self.kmeans_init:
                kmeans_start = time.time()
                print("Started K-means color extraction from the target image. Computing {:d} clusters.\n".format(self.kmeans_clusters))
                color_palette = kmeans_color_palette(self.target, self.kmeans_clusters, 
                                                     self.kmeans_iterations, self.kmeans_repeats)
                kmeans_end = time.time()

            else:
                color_palette = None
            
            print("K-means finished in {:.2f} seconds.".format(kmeans_end - kmeans_start))
            self.population.generate_initial_population(color_palette)
            print("Finished generating initial population.\n")

        else:
            print("Continuing evolution from cached data.")

        for i in range(self.last_generation_index, self.generations):
            s = time.time()
            self.generation_change()
            e = time.time()

            print("Generation: {:d}".format(i))
            print("Best global fitness: {:.5f}".format(self.global_best_f))
            print("Time taken: {:.2f}\n".format(e - s))

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
                "kmeans_init": self.kmeans_init,
                "kmeans_clusters": self.kmeans_clusters,
                "kmeans_iterations": self.kmeans_iterations,
                "kmeans_repeats": self.kmeans_repeats,
                "last_generation_index": generation_index,
                "caching_ratio_period": self.caching_ratio_period,
                "fittest_survival_ratio": self.fittest_survival_ratio,
                "fitness_label": self.fitness_label,
                "crossover_label": self.crossover_label,
                "population_label": self.population.__class__.__name__
            }
            json.dump(d, f, indent = 4)

        self.population.export("./{}".format(self.alg_label))
        history_path = "./{}/fitness_history.txt".format(self.alg_label)
        with open(history_path, "a") as f:
            f.write("{:f}\n".format(self.global_best_f))

        if generation_index % int(self.caching_ratio_period * self.generations) == 0 or\
           generation_index == self.generations - 1:
           with \
                open("./{}/best_chromosomes_history/{:d}.npy".format(self.alg_label, generation_index), "wb")\
                as f:
                np.save(f, self.global_best)

    @staticmethod
    def load(root):
        with open("./{}/evolver_metadata.json".format(root), "r") as f:
            data = json.load(f)

        if data["population_label"] == "PopulationPolyRBG":
            population = PopulationPolyRBG.load(root)
        
        target = cv2.imread(data["target_path"])
        res = GAEvolver(root, target, data["target_path"], data["generations"], 
                       data["fittest_survival_ratio"], population, data["fitness_label"], 
                       data["crossover_label"], data["kmeans_init"], 
                       kmeans_clusters = data["kmeans_clusters"], kmeans_iterations = data["kmeans_iterations"],
                       kmeans_repeats = data["kmeans_repeats"], caching = True, 
                       caching_ratio_period = data["caching_ratio_period"])

        res.last_generation_index = data["last_generation_index"] + 1

        print("Finished loading the cached GAEvolver object.\n")
        return res

    def paint_canvas(self, generation_index):
        with open("./{}/best_chromosomes_history/{:d}.npy".format(self.alg_label, generation_index), "rb") as f:
            chromosome = np.load(f)
        
        canvas = self.population.embed_chromosome(chromosome)
        cv2.imshow("./{}-{:d}".format(self.alg_label, generation_index), canvas)
        cv2.waitKey(0)