import numpy as np
import time
import cv2
from typing import List

#GLOBALS
NUMBER_OF_ELLIPSES = 150
POPULATION_SIZE = 100
WIDTH = 600
HEIGHT = 450
GENERATIONS = 10
MUTATION_RATE = 0.001
FITTEST_SURVIVAL_RATIO = 0.5

FEATURE_SPACE = NUMBER_OF_ELLIPSES * 5

GLOBAL_BEST = None
GLOBAL_BEST_F = None

# One ellipse - center, axes, color -> 5 parameters

def fitness(target : np.ndarray, image : np.ndarray) -> float:
    # return 1 - 1 / target.size * ((target - image) ** 2).sum()
    return cv2.PSNR(target, image)

def generate_initial_population(rand : np.random.Generator):
    res = np.empty((POPULATION_SIZE, NUMBER_OF_ELLIPSES, 5), dtype = np.int32)

    res[:, :, 0] = rand.integers(0, 
                                 WIDTH, 
                                 size = (POPULATION_SIZE, NUMBER_OF_ELLIPSES),
                                 dtype = np.int32)

    res[:, :, 1] = rand.integers(0, 
                                 HEIGHT, 
                                 size = (POPULATION_SIZE, NUMBER_OF_ELLIPSES),
                                 dtype = np.int32)
    
    x_coords = res[:, :, 0]
    y_coords = res[:, :, 1]
    
    res[:, :, 2] = rand.integers(0,
                                 np.minimum(x_coords, WIDTH - x_coords) + 1,
                                 dtype = np.int32)

    res[:, :, 3] = rand.integers(0,
                                np.minimum(y_coords, HEIGHT - y_coords) + 1,
                                dtype = np.int32)

    res[:, :, 4] = rand.integers(0, 
                                 256, 
                                 size = (POPULATION_SIZE, NUMBER_OF_ELLIPSES),
                                 dtype = np.int32)

    # Precompute drawings
    drawings = np.empty((POPULATION_SIZE, HEIGHT, WIDTH), dtype = np.uint8)
    for i in range(POPULATION_SIZE):
        drawings[i] = embed_chromosome(res[i])
    
    return res, drawings

def embed_chromosome(chromosome : np.ndarray):
    canvas = np.full((HEIGHT, WIDTH), 255, dtype = np.uint8)
    canvas_copy = np.full((HEIGHT, WIDTH), 255, dtype = np.uint8)
    for i in range(NUMBER_OF_ELLIPSES):
        vertices = tuple(chromosome[i, [0, 1]])
        axes = tuple(chromosome[i, [2, 3]])
        color = int(chromosome[i, 4])
        cv2.ellipse(canvas_copy, vertices, axes, 0, 0, 360, color, -1)
        cv2.addWeighted(canvas_copy, 0.5, canvas, 0.5, 0, canvas)
    return canvas

def crossover_uniform(x : np.ndarray, y : np.ndarray, rand : np.random.Generator, fx, fy):
    c1 = np.empty((NUMBER_OF_ELLIPSES, 5), dtype = np.int32)
    c2 = np.empty((NUMBER_OF_ELLIPSES, 5), dtype = np.int32)

    fx_prime = fx + (1 - fx - fy) / 2
    fy_prime = fy + (1 - fx - fy) / 2
    choices = [0, 1]
    
    # Polygon and color selection
    for i in range(NUMBER_OF_ELLIPSES):
        cx_e = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cx_a = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cx_c = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

        cy_e = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cy_a = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
        cy_c = rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

        c1[i, [0, 1]] = x[i, [0, 1]] if cx_e == 0 else y[i, [0, 1]]
        c1[i, [2, 3]] = x[i, [2, 3]] if cx_a == 0 else y[i, [2, 3]]
        c1[i, 4] = x[i, 4] if cx_c == 0 else y[i, 4]

        c2[i, [0, 1]] = x[i, [0, 1]] if cy_e == 0 else y[i, [0, 1]]
        c2[i, [2, 3]] = x[i, [2, 3]] if cy_a == 0 else y[i, [2, 3]]
        c2[i, 4] = x[i, 4] if cy_c == 0 else y[i, 4]

    return c1, c2

def mutate(chromosome : np.ndarray, rand : np.random.Generator):
    mask_coords = np.ones((NUMBER_OF_ELLIPSES, 5), dtype = np.float32)
    mask_axes = np.ones((NUMBER_OF_ELLIPSES, 5), dtype = np.float32)
    mask_color = np.ones((NUMBER_OF_ELLIPSES, 5), dtype = np.float32)

    mask_coords[:, :2] = rand.uniform(size = (NUMBER_OF_ELLIPSES, 2))
    mask_axes[:, 2:4] = rand.uniform(size = (NUMBER_OF_ELLIPSES, 2))
    mask_color[:, 4] = rand.uniform(size = (NUMBER_OF_ELLIPSES))

    mask_coords = mask_coords < MUTATION_RATE
    mask_axes = mask_axes < MUTATION_RATE
    mask_color = mask_color < MUTATION_RATE

    mutations_coords = mask_coords.sum().sum()
    mutations_axes = mask_axes.sum().sum()
    mutations_color = mask_color.sum().sum()

    if mutations_coords > 0:
        chromosome[mask_coords] = rand.integers(0, WIDTH, size = mutations_coords, dtype = np.int32)
    
    if mutations_axes > 0:
        chromosome[mask_axes] = rand.integers(0, 
                                                   min(WIDTH, HEIGHT), 
                                                   size = mutations_axes, dtype = np.int32)
    if mutations_color > 0:
        chromosome[mask_color] = rand.integers(0, 256, size = mutations_color, dtype = np.int32)

    return chromosome

def generation_change(population : np.ndarray, drawings: np.ndarray, 
                      target : np.ndarray, rand : np.random.Generator):

    global GLOBAL_BEST, GLOBAL_BEST_F
    target_norm = target.astype(np.float64) / 255.0
    f_values = np.empty(POPULATION_SIZE, dtype = np.float64)

    # Fitness calculation
    for i in range(POPULATION_SIZE):
        f_values[i] = fitness(target_norm, drawings[i].astype(np.float64) / 255.0)

    indices = np.argsort(-f_values)
    
    # Update the best global configuration
    if GLOBAL_BEST_F is None or GLOBAL_BEST_F < f_values[indices[0]]:
        GLOBAL_BEST = population[indices[0]]
        GLOBAL_BEST_F = f_values[indices[0]] 

    new_population = np.empty((POPULATION_SIZE, NUMBER_OF_ELLIPSES, 5),
                               dtype = np.int32)
    new_drawings = np.empty((POPULATION_SIZE, HEIGHT, WIDTH), dtype = np.uint8)

    # Survival of the fittest
    fittest_threshold = int(POPULATION_SIZE * FITTEST_SURVIVAL_RATIO)
    indices = indices[:fittest_threshold]
    fsum = f_values[indices].sum()
    categorical_weights = np.array([item / fsum for item in f_values[indices]], dtype = np.float64)
    new_population[:fittest_threshold] = population[indices]
    new_drawings[:fittest_threshold] = drawings[indices]

    # Selection
    size = fittest_threshold
    while size < POPULATION_SIZE:
        parrent_indices = rand.choice(indices, size = 2, p = categorical_weights)
        # c1, c2 = crossover(population[parrent_indices[0], :], population[parrent_indices[1], :], rand)
        c1, c2 = crossover_uniform(population[parrent_indices[0]], 
                                   population[parrent_indices[1]], 
                                   rand, 
                                   f_values[parrent_indices[0]] / fsum, 
                                   f_values[parrent_indices[1]] / fsum)
        new_population[size, :] = c1
        new_population[size + 1, :] = c2
        size += 2
    
    # Mutation, every gene of offsprings mutates with small probability
    for i in range(fittest_threshold, POPULATION_SIZE):
        new_population[i] = mutate(new_population[i], rand)
        drawing = embed_chromosome(new_population[i])
        new_drawings[i] = drawing

    return new_population, new_drawings

def evolution(target : np.ndarray, rand : np.random.Generator):
    population, drawings = generate_initial_population(rand)
    print("Finished generating initial population.\n")
    for i in range(GENERATIONS):
        s = time.time()
        population, drawings = generation_change(population, drawings, target, rand)
        e = time.time()
        print("Generation: {:d}".format(i + 1))
        print("Best global fitness: {:.2f}".format(GLOBAL_BEST_F))
        print("Time taken: {:.2f}".format(e - s))
    
    res = embed_chromosome(GLOBAL_BEST)
    cv2.imshow("Result", res)
    cv2.waitKey(0)

if __name__ == "__main__":
    cv2.setRNGSeed(41)
    rand = np.random.default_rng(41)

    target = cv2.imread("moon.jpeg")
    WIDTH = target.shape[1]
    HEIGHT = target.shape[0]

    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    evolution(target, rand)