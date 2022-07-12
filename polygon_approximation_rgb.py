import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import json
import os

from typing import List
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

#GLOBALS
NUMBER_OF_VERTICES = 3
NUMBER_OF_POLYGONS = 5
POPULATION_SIZE = 200
WIDTH = None
HEIGHT = None
GENERATIONS = 100
EXPECTED_NUMBER_OF_MUTATIONS = 4 # Idea - keep mutation rate high at the begining but lower it
                                 # as the algorithm progresses
MUTATION_RATE = EXPECTED_NUMBER_OF_MUTATIONS / (NUMBER_OF_POLYGONS * (2 * NUMBER_OF_VERTICES + 3))
FITTEST_SURVIVAL_RATIO = 0.5

CROSSOVER_CUTS = [1, 2, 3, 4, 5] # Number of crossover points to be chosen
CROSSOVER_CUT_RANGE = list(range(1, NUMBER_OF_POLYGONS + 1)) # Precomputed list of cuts for cutting crossover

KMEANS_CLUSTERS = 10
KMEANS_ITERATIONS = 100
KMEANS_REPEATS = 10

GLOBAL_BEST = None
GLOBAL_BEST_F = None

def kmeans_color_palette(image : np.ndarray):
    input = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, KMEANS_ITERATIONS, -1)
    c, l, centers = cv2.kmeans(input, KMEANS_CLUSTERS, None, criteria, KMEANS_REPEATS, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.int32)
    return centers

def fitness_psnr(target : np.ndarray, image : np.ndarray) -> float:
    return cv2.PSNR(target, image)

def fitness_nmse(target: np.ndarray, image : np.ndarray) -> float:
    # return - ((target - image) ** 2).sum() / (WIDTH * HEIGHT * 3)
    return -mse(target, image)

def fitness_ssim(target : np.ndarray, image : np.ndarray) -> float:
    return ssim(target, image, channel_axis = 2)
    # channel_axis = 2 for RGB images, should be removed for grayscale inputs

def generate_initial_population(rand : np.random.Generator, color_palette: np.ndarray):
    res = np.empty((POPULATION_SIZE, NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES * 2 + 3), dtype = np.int32)

    res[:, :, :NUMBER_OF_VERTICES * 2:2] = rand.integers(0, 
                                            WIDTH, 
                                            size = (POPULATION_SIZE, NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES),
                                            dtype = np.int32)

    res[:, :, 1:NUMBER_OF_VERTICES * 2:2] = rand.integers(0, 
                                            HEIGHT, 
                                            size = (POPULATION_SIZE, NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES),
                                            dtype = np.int32)
    
    color_indices = rand.choice(color_palette.shape[0], size = POPULATION_SIZE * NUMBER_OF_POLYGONS, replace = True)
    res[:, :, NUMBER_OF_VERTICES * 2:] = color_palette[color_indices, :].reshape(POPULATION_SIZE, NUMBER_OF_POLYGONS, 3)

    # Precompute drawings
    drawings = np.empty((POPULATION_SIZE, HEIGHT, WIDTH, 3), dtype = np.uint8)
    for i in range(POPULATION_SIZE):
        drawings[i] = embed_chromosome(res[i])
    
    return res, drawings

# Specify correct input type
def embed_chromosome(chromosome : np.ndarray):
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype = np.uint8)
    canvas_copy = np.full((HEIGHT, WIDTH, 3), 255, dtype = np.uint8)
    for i in range(NUMBER_OF_POLYGONS):
        vertices = chromosome[i, :2 * NUMBER_OF_VERTICES].reshape(1, -1, 2)
        color = tuple(chromosome[i, 2 * NUMBER_OF_VERTICES:].tolist())
        cv2.fillPoly(canvas_copy, [vertices], color)
        cv2.addWeighted(canvas_copy, 0.5, canvas, 0.5, 0, canvas)
    return canvas
        
def crossover_cuts(x : np.ndarray, y : np.ndarray, rand : np.random.Generator):
    # [1, 5] crossover points are chosen
    # Colors of copied polygons remain the same, for now.

    c1 = np.empty(x.shape, dtype = np.int32)
    c2 = np.empty(x.shape, dtype = np.int32)
    l = NUMBER_OF_POLYGONS
    num_cuts = rand.choice(CROSSOVER_CUTS)
    cuts = [l]

    # Evade the edge case of entirely copying parrents
    while len(cuts) == 1 and cuts[0] == l:
        cuts = rand.choice(CROSSOVER_CUT_RANGE, size = num_cuts, replace = False)

    cuts = np.sort(cuts)

    left, counter = 0, 0
    for right in cuts:
        if counter & 1:
            c1[left:right] = x[left:right].copy()
            c2[left:right] = y[left:right].copy()
        else:
            c1[left:right] = y[left:right].copy()
            c2[left:right] = x[left:right].copy()

        counter += 1
        left = right
    
    if left < l:
        if counter & 1:
            c1[left:l] = x[left:l].copy()
            c2[left:l] = y[left:l].copy()
        else:
            c1[left:l] = y[left:l].copy()
            c2[left:l] = x[left:l].copy()

    return c1, c2

def crossover_uniform(x : np.ndarray, y : np.ndarray, rand : np.random.Generator, fx, fy):
    """
    Uniform crossover weighted by fvalues of parrents.
    """
    c1 = np.empty((NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES * 2 + 3), dtype = np.int32)
    c2 = np.empty((NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES * 2 + 3), dtype = np.int32)

    fx_prime = fx + (1 - fx - fy) / 2
    fy_prime = fy + (1 - fx - fy) / 2
    choices = [0, 1]
    
    # Polygon and color selection
    for i in range(NUMBER_OF_POLYGONS):
        cx_p = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cy_p = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

        cx_c = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cy_c = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

        c1[i, :2 * NUMBER_OF_VERTICES] = x[i, :2 * NUMBER_OF_VERTICES].copy() if cx_p == 0 else y[i, :2 * NUMBER_OF_VERTICES].copy()
        c2[i, :2 * NUMBER_OF_VERTICES] = x[i, :2 * NUMBER_OF_VERTICES].copy() if cy_p == 0 else y[i, :2 * NUMBER_OF_VERTICES].copy()

        c1[i, 2 * NUMBER_OF_VERTICES:] = x[i, 2 * NUMBER_OF_VERTICES:].copy() if cx_c == 0 else y[i, 2 * NUMBER_OF_VERTICES:].copy()
        c2[i, 2 * NUMBER_OF_VERTICES:] = x[i, 2 * NUMBER_OF_VERTICES:].copy() if cy_c == 0 else y[i, 2 * NUMBER_OF_VERTICES:].copy()

    return c1, c2

def mutate(chromosome : np.ndarray, rand : np.random.Generator):
    mutation_matrix = rand.uniform(size = (NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES * 2 + 3))
    
    mutations_x = np.sum(mutation_matrix[:, :NUMBER_OF_VERTICES * 2:2] < MUTATION_RATE)
    mutations_y = np.sum(mutation_matrix[:, 1:NUMBER_OF_VERTICES * 2:2] < MUTATION_RATE)
    mutations_rgb = np.sum(mutation_matrix[:, NUMBER_OF_VERTICES * 2:] < MUTATION_RATE)

    if mutations_x > 0:
        chromosome[:, :NUMBER_OF_VERTICES * 2:2]\
            [mutation_matrix[:, :NUMBER_OF_VERTICES * 2:2] < MUTATION_RATE] = \
            rand.integers(0, WIDTH, size = (mutations_x), dtype = np.int32)

    if mutations_y > 0:
        chromosome[:, 1:NUMBER_OF_VERTICES * 2:2]\
        [mutation_matrix[:, 1:NUMBER_OF_VERTICES * 2:2] < MUTATION_RATE] = \
        rand.integers(0, HEIGHT, size = (mutations_y), dtype = np.int32)

    if mutations_rgb > 0:
        chromosome[:, NUMBER_OF_VERTICES * 2:]\
        [mutation_matrix[:, NUMBER_OF_VERTICES * 2:] < MUTATION_RATE] = \
        rand.integers(0, 256, size = (mutations_rgb), dtype = np.int32)

    return chromosome

def generation_change(population : np.ndarray, drawings: np.ndarray, 
                      target : np.ndarray, rand : np.random.Generator, fitness):

    global GLOBAL_BEST, GLOBAL_BEST_F
    target_norm = target.astype(np.float64) / 255.0
    f_values = np.empty(POPULATION_SIZE, dtype = np.float64)

    # Fitness calculation
    for i in range(POPULATION_SIZE):
        f_values[i] = fitness(target_norm, drawings[i].astype(np.float64) / 255.0)

    indices = np.argsort(-f_values)
    
    # Update the best global configuration
    if GLOBAL_BEST_F is None or GLOBAL_BEST_F < f_values[indices[0]]:
        GLOBAL_BEST = population[indices[0]].copy()
        GLOBAL_BEST_F = f_values[indices[0]]

    new_population = np.empty((POPULATION_SIZE, NUMBER_OF_POLYGONS, NUMBER_OF_VERTICES * 2 + 3),
                               dtype = np.int32)
    new_drawings = np.empty((POPULATION_SIZE, HEIGHT, WIDTH, 3), dtype = np.uint8)

    # Survival of the fittest
    fittest_threshold = int(POPULATION_SIZE * FITTEST_SURVIVAL_RATIO)
    indices = indices[:fittest_threshold]
    fsum = f_values[indices].sum()
    categorical_weights = np.array([item / fsum for item in f_values[indices]], dtype = np.float64)
    new_population[:fittest_threshold] = population[indices]
    new_drawings[:fittest_threshold] = drawings[indices]

    # Selection and mating
    size = fittest_threshold
    while size < POPULATION_SIZE:
        parrent_indices = rand.choice(indices, size = 2, p = categorical_weights)
        c1, c2 = crossover_cuts(population[parrent_indices[0]], population[parrent_indices[1]], rand)
        '''
        c1, c2 = crossover_uniform(population[parrent_indices[0]], 
                                   population[parrent_indices[1]], 
                                   rand, 
                                   f_values[parrent_indices[0]] / fsum, 
                                   f_values[parrent_indices[1]] / fsum)
        '''
        new_population[size] = c1
        new_population[size + 1] = c2
        size += 2
    
    # Mutation, every gene of offsprings mutates with small probability
    for i in range(fittest_threshold, POPULATION_SIZE):
        new_population[i] = mutate(new_population[i], rand)
        drawing = embed_chromosome(new_population[i])
        new_drawings[i] = drawing

    return new_population, new_drawings

def evolution(target : np.ndarray, color_palette : np.ndarray, rand : np.random.Generator,
              fitness, alg_label : str, caching : bool, cache_period : int):
              
    population, drawings = generate_initial_population(rand, color_palette)
    print("Finished generating initial population.\n")

    fitness_history = []
    for i in range(GENERATIONS):
        s = time.time()
        population, drawings = generation_change(population, drawings, target, rand, fitness)
        fitness_history.append(GLOBAL_BEST_F)
        e = time.time()

        print("Generation: {:d}".format(i))
        print("Best global fitness: {:.5f}".format(GLOBAL_BEST_F))
        print("Time taken: {:.2f}\n".format(e - s))

        if caching:
            with open("./{}/last_generation.npy".format(alg_label), "wb") as f:
                np.save(f, population)

            with open("./{}/metadata.json".format(alg_label), "r+") as f:
                metadata = json.load(f)
                metadata["last_generation_index"] = i
                f.truncate(0)
                f.seek(0)
                json.dump(metadata, f, indent = 4)

            if i % cache_period == 0 or i == GENERATIONS:
                with open("./{}/best_chromosomes_history/{:d}.npy".format(alg_label, i), "wb") as f:
                    np.save(f, GLOBAL_BEST)
                
                with open("./{}/fitness_history.npy".format(alg_label), "wb") as f:
                    np.save(f, np.array(fitness_history))

def evolve_image(name : str, seed : int, alg_label : str, fitness_label : str,
                 caching = False, cache_period : int = 100):
    global WIDTH, HEIGHT
    rand = np.random.default_rng(seed)
    cv2.setRNGSeed(seed)
    image_path = "./images/{}".format(name)

    target = cv2.imread(image_path)
    WIDTH = target.shape[1]
    HEIGHT = target.shape[0]

    if caching:
        if not os.path.exists("./{}".format(alg_label)):
            os.mkdir("./{}".format(alg_label))
            os.mkdir("./{}/best_chromosomes_history".format(alg_label))

        with open("./{}/metadata.json".format(alg_label), "w+") as f:
            d = {
                "target_path_root_relative": image_path,
                "number_of_vertices": NUMBER_OF_VERTICES,
                "number_of_polygons": NUMBER_OF_POLYGONS,
                "population_size": POPULATION_SIZE,
                "fitness_label": fitness_label,
                "width": WIDTH,
                "height": HEIGHT,
                "generations": GENERATIONS,
                "exp_number_of_mutations": EXPECTED_NUMBER_OF_MUTATIONS,
                "last_generation_index": 0
            }

            json.dump(d, f, indent = 4)
    
    color_palette = kmeans_color_palette(target)
    print("Finished KMeans color palette generation for {:d} clusters.\n".format(KMEANS_CLUSTERS))
    evolution(target, color_palette, rand, fitness_psnr, alg_label, caching, cache_period)

if __name__ == "__main__":
    evolve_image("poly_benchmark_0.jpg", 2, "test", "PSNR",
                  caching = True, cache_period = 10)