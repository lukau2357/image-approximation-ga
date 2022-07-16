import numpy as np
import cv2
import os
import json

from population import Population

class PopulationPolyRBG(Population):
    def __init__(self, number_of_vertices : int, number_of_polygons : int, population_size : int,
                width : int, height : int, expected_number_of_mutations : int, rand_seed : int):

        super().__init__(width, height, population_size, expected_number_of_mutations, rand_seed)

        self.number_of_vertices = number_of_vertices
        self.number_of_polygons = number_of_polygons
        self.feature_space_dim = number_of_polygons * (2 * number_of_vertices + 3)
        self.mutation_rate = expected_number_of_mutations / self.feature_space_dim
        self.crossover_cuts_number = [1, 2, 3, 4, 5]
        self.crossover_cuts_range = list(range(1, self.number_of_polygons + 1))

        self.current_generation = np.empty((self.population_size, 
                                    self.number_of_polygons, 
                                    2 * self.number_of_vertices + 3), 
                                    dtype = np.int32)

        self.drawings = np.empty((self.population_size, 
                                  self.height, 
                                  self.width, 3), dtype = np.uint8)
        
    def embed_chromosome(self, chromosome : np.ndarray):
        canvas = np.full((self.height, self.width, 3), 255, dtype = np.uint8)
        canvas_copy = np.full((self.height, self.width, 3), 255, dtype = np.uint8)
        for i in range(self.number_of_polygons):
            vertices = chromosome[i, :2 * self.number_of_vertices].reshape(1, -1, 2)
            color = tuple(chromosome[i, 2 * self.number_of_vertices:].tolist())
            cv2.fillPoly(canvas_copy, [vertices], color)
            cv2.addWeighted(canvas_copy, 0.5, canvas, 0.5, 0, canvas)
        return canvas

    def generate_initial_population(self, color_palette: np.ndarray = None):
        res = np.empty((self.population_size, 
                        self.number_of_polygons,
                        self.number_of_vertices * 2 + 3), dtype = np.int32)

        res[:, :, :self.number_of_vertices * 2:2] = self.rand.integers(0, 
                                                self.width, 
                                                size = (self.population_size, 
                                                        self.number_of_polygons, 
                                                        self.number_of_vertices),
                                                dtype = np.int32)

        res[:, :, 1:self.number_of_vertices * 2:2] = self.rand.integers(0, 
                                                self.height, 
                                                size = (self.population_size, 
                                                        self.number_of_polygons, 
                                                        self.number_of_vertices),
                                                dtype = np.int32)
        
        color_indices = self.rand.choice(color_palette.shape[0], 
                                    size = self.population_size * self.number_of_polygons, 
                                    replace = True)
        res[:, :, self.number_of_vertices * 2:] = \
            color_palette[color_indices, :].reshape(self.population_size, 
                                                    self.number_of_polygons, 3)

        # Precompute drawings
        drawings = np.empty((self.population_size, self.height, self.width, 3), 
                            dtype = np.uint8)
        for i in range(self.population_size):
            drawings[i] = self.embed_chromosome(res[i])
        
        self.current_generation = res.copy()
        self.drawings = drawings

    def crossover_cuts(self, *args):
        x = args[0]
        y = args[1]

        c1 = np.empty(x.shape, dtype = np.int32)
        c2 = np.empty(x.shape, dtype = np.int32)
        l = self.number_of_polygons
        num_cuts = self.rand.choice(self.crossover_cuts_number)
        cuts = [l]

        # Evade the edge case of entirely copying parrents
        while len(cuts) == 1 and cuts[0] == l:
            cuts = self.rand.choice(self.crossover_cuts_range, 
                               size = num_cuts, 
                               replace = False)

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

    def crossover_uniform(self, *args):
        """
        Uniform crossover weighted by fvalues of parrents.
        """
        x = args[0]
        y = args[1]
        fx = args[2]
        fy = args[3]

        c1 = np.empty((self.number_of_polygons, self.number_of_vertices * 2 + 3), 
                        dtype = np.int32)
        c2 = np.empty((self.number_of_polygons, self.number_of_vertices * 2 + 3), dtype = np.int32)

        fx_prime = fx + (1 - fx - fy) / 2
        fy_prime = fy + (1 - fx - fy) / 2
        choices = [0, 1]
        
        # Polygon and color selection
        for i in range(self.number_of_polygons):
            cx_p = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
            cy_p = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

            cx_c = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
            cy_c = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

            c1[i, :2 * self.number_of_vertices] = x[i, :2 * self.number_of_vertices].copy() if cx_p == 0 else y[i, :2 * self.number_of_vertices].copy()
            c2[i, :2 * self.number_of_vertices] = x[i, :2 * self.number_of_vertices].copy() if cy_p == 0 else y[i, :2 * self.number_of_vertices].copy()

            c1[i, 2 * self.number_of_vertices:] = x[i, 2 * self.number_of_vertices:].copy() if cx_c == 0 else y[i, 2 * self.number_of_vertices:].copy()
            c2[i, 2 * self.number_of_vertices:] = x[i, 2 * self.number_of_vertices:].copy() if cy_c == 0 else y[i, 2 * self.number_of_vertices:].copy()

        return c1, c2

    def mutate(self, chromosome : np.ndarray):
        mutation_matrix = self.rand.uniform(size = (self.number_of_polygons, 
                                                        self.number_of_vertices * 2 + 3))
        
        mutations_x = np.sum(mutation_matrix[:, :self.number_of_vertices * 2:2] < self.mutation_rate)
        mutations_y = np.sum(mutation_matrix[:, 1:self.number_of_vertices * 2:2] < self.mutation_rate)
        mutations_rgb = np.sum(mutation_matrix[:, self.number_of_vertices * 2:] < self.mutation_rate)

        if mutations_x > 0:
            chromosome[:, :self.number_of_vertices * 2:2]\
                [mutation_matrix[:, :self.number_of_vertices * 2:2] < self.mutation_rate] = \
                self.rand.integers(0, self.width, size = (mutations_x), dtype = np.int32)

        if mutations_y > 0:
            chromosome[:, 1:self.number_of_vertices * 2:2]\
            [mutation_matrix[:, 1:self.number_of_vertices * 2:2] < self.mutation_rate] = \
            self.rand.integers(0, self.height, size = (mutations_y), dtype = np.int32)

        if mutations_rgb > 0:
            chromosome[:, self.number_of_vertices * 2:]\
            [mutation_matrix[:, self.number_of_vertices * 2:] < self.mutation_rate] = \
            self.rand.integers(0, 256, size = (mutations_rgb), dtype = np.int32)

    def mutate_generation(self, mutation_threshold):
        mutation_tensor = self.rand.uniform(size = (self.population_size - mutation_threshold,
                                                    self.number_of_polygons,
                                                    self.number_of_vertices * 2 + 3))

        mutations_x = np.sum(mutation_tensor[:, :, :self.number_of_vertices * 2:2] < self.mutation_rate)
        mutations_y = np.sum(mutation_tensor[:, :, 1:self.number_of_vertices * 2:2] < self.mutation_rate)
        mutations_rgb = np.sum(mutation_tensor[:, :, self.number_of_vertices * 2:] < self.mutation_rate)

        if mutations_x > 0:
            self.current_generation[mutation_threshold:, :, :self.number_of_vertices * 2:2]\
                [mutation_tensor[:, :, :self.number_of_vertices * 2:2] < self.mutation_rate] = \
                self.rand.integers(0, self.width, size = (mutations_x), dtype = np.int32)

        if mutations_y > 0:
            self.current_generation[mutation_threshold:, :, 1:self.number_of_vertices * 2:2]\
            [mutation_tensor[:, :, 1:self.number_of_vertices * 2:2] < self.mutation_rate] = \
            self.rand.integers(0, self.height, size = (mutations_y), dtype = np.int32)

        if mutations_rgb > 0:
            self.current_generation[mutation_threshold:, :, self.number_of_vertices * 2:]\
            [mutation_tensor[:, :, self.number_of_vertices * 2:] < self.mutation_rate] = \
            self.rand.integers(0, 256, size = (mutations_rgb), dtype = np.int32)

    def export(self, root):
        metadata_path = "{}/population_metadata.json".format(root)
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                d = {
                    "width": self.width,
                    "height": self.height,
                    "population_size": self.population_size,
                    "expected_no_mutations": self.expected_number_of_mutations,
                    "rand_seed" : self.rand_seed,
                    "number_of_vertices": self.number_of_vertices,
                    "number_of_polygons": self.number_of_polygons,
                    "feature_space_dim" : self.feature_space_dim,
                    "mutation_rate": self.mutation_rate
                }

                json.dump(d, f, indent = 4)
        
        last_generation_path = "{}/last_generation.npy".format(root)
        with open(last_generation_path, "wb") as f:
            np.save(f, self.current_generation)
    
    @staticmethod
    def load(root):
        with open("./{}/population_metadata.json".format(root), "r") as f:
            data = json.load(f)

        res = PopulationPolyRBG(data["number_of_vertices"], data["number_of_polygons"],
        data["population_size"], data["width"], data["height"], data["expected_no_mutations"],
        data["rand_seed"])

        with open("./{}/last_generation.npy".format(root), "rb") as f:
            res.current_generation = np.load(f)
            for i in range(res.population_size):
                res.drawings[i] = res.embed_chromosome(res.current_generation[i])

        print("Finished loading the cached Population object.")
        return res