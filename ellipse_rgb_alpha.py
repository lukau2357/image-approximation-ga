import numpy as np
import cv2
import json
import os

from population import Population

'''
cv2.ellipse documentation - https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
'''

class PopulationEllipseRGBA(Population):
    def __init__(self, number_of_ellipses : int, population_size : int, width : int, height : int,
                 expected_number_of_mutations : int, rand_seed : int):

        super().__init__(width, height, population_size, expected_number_of_mutations, rand_seed)
        self.number_of_ellipses = number_of_ellipses

        # center_x, center_y, axis_x, axis_y, rotation_angle, r, g, b, alpha
        self.feature_space_dim = self.number_of_ellipses * 9
        self.mutation_rate = self.expected_number_of_mutations / self.feature_space_dim

        self.crossover_cuts_number = [1, 2, 3, 4, 5]
        self.crossover_cuts_range = list(range(1, self.number_of_ellipses + 1))

        self.current_generation = np.empty((self.population_size, self.number_of_ellipses, 9), dtype = np.float64)
        self.drawings = np.empty((self.population_size, self.height, self.width, 3), dtype = np.uint8)

    def embed_chromosome(self, chromosome : np.ndarray):
        canvas = np.zeros((self.height, self.width, 3), dtype = np.uint8)
        canvas_copy = canvas.copy()
        for i in range(self.number_of_ellipses):
            center = tuple(chromosome[i, :2].astype(np.int32))
            axes = tuple(chromosome[i, 2:4].astype(np.int32))
            rotation_angle = int(chromosome[i, 4])
            color = tuple(chromosome[i, 5:8].astype(np.int32).tolist())
            alpha = chromosome[i, 8]
            cv2.ellipse(canvas_copy, center, axes, rotation_angle, 0, 360, color, -1)
            cv2.addWeighted(canvas_copy, alpha, canvas, 1 - alpha, 0, canvas)
            cv2.ellipse(canvas_copy, center, axes, rotation_angle, 0, 360, (0, 0, 0), -1)
        return canvas

    def generate_initial_population(self, target : np.ndarray):
        res = np.empty((self.population_size, self.number_of_ellipses, 9))
        res[:, :, 0] = self.rand.integers(0, self.width, size = 
                                          (self.population_size, self.number_of_ellipses))
        res[:, :, 1] = self.rand.integers(1, self.height, size = 
                                          (self.population_size, self.number_of_ellipses))

        res[:, :, 2] = self.rand.integers(np.maximum(res[:, :, 0], self.width - res[:, :, 0]), size = 
                                          (self.population_size, self.number_of_ellipses))

        res[:, :, 3] = self.rand.integers(np.maximum(res[:, :, 1], self.height - res[:, :, 1]), size = 
                                          (self.population_size, self.number_of_ellipses))

        res[:, :, 4] = self.rand.integers(0, 360, size = 
                                          (self.population_size, self.number_of_ellipses))
        
        x_means = res[:, :, 0].mean(axis = 2).astype(np.int32)
        y_means = res[:, :, 1].mean(axis = 2).astype(np.int32)
        res[:, :, 5:8] = target[y_means, x_means]

        res[:, :, 8] = self.rand.uniform(size = (self.population_size, self.number_of_ellipses))

        self.current_generation = res
        self.drawings = np.empty((self.population_size, self.height, self.width, 3), dtype = np.uint8)

        for i in range(self.population_size):
            self.drawings[i] = self.embed_chromosome(res[i])
    
    def crossover_cuts(self, *args):
        x = args[0]
        y = args[1]

        c1 = np.empty(x.shape, dtype = np.float64)
        c2 = np.empty(x.shape, dtype = np.float64)
        l = self.number_of_ellipses
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

        c1 = np.empty(x.shape, dtype = np.float64)
        c2 = np.empty(x.shape, dtype = np.float64)

        fx_prime = fx + (1 - fx - fy) / 2
        fy_prime = fy + (1 - fx - fy) / 2
        choices = [0, 1]
        
        for i in range(self.number_of_ellipses):
            cx_p = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
            cy_p = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

            cx_c = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]
            cy_c = self.rand.choice(choices, size = 1, p = [fx_prime, fy_prime])[0]

            c1[i, 0:5] = x[i, 0:5].copy() if cx_p == 0 else y[i, 0:5].copy()
            c2[i, 0:5] = x[i, 0:5].copy() if cy_p == 0 else y[i, 0:5].copy()

            c1[i, 5:8] = x[i, 5:8].copy() if cx_c == 0 else y[i, 5:8].copy()
            c2[i, 5:8] = x[i, 5:8].copy() if cy_c == 0 else y[i, 5:8].copy()

        betas = self.rand.uniform(size = (self.number_of_ellipses))
        c1[:, 8] = betas * x[:, 8] + (1 - betas) * y[:, 8]
        c2[:, 8] = betas * y[:, 8] + (1 - betas) * x[:, 8]

        return c1, c2

    def mutate(self, chromosome : np.ndarray):
        mutation_matrix = self.rand.uniform(size = (self.number_of_ellipses, 9))
        
        xaxes_mask = mutation_matrix[:, [0, 2]] < self.mutation_rate
        yaxes_mask = mutation_matrix[:, [1, 3]] < self.mutation_rate
        rotangle_mask = mutation_matrix[:, 4] < self.mutation_rate
        color_mask = mutation_matrix[:, 5:8] < self.mutation_rate
        alpha_mask = mutation_matrix[:, 8] < self.mutation_rate

        mutations_xaxes = np.sum(xaxes_mask)
        mutations_yaxes = np.sum(yaxes_mask)
        mutations_rotangle = np.sum(rotangle_mask)
        mutations_color = np.sum(color_mask)
        mutations_alpha = np.sum(alpha_mask)
        
        if mutations_xaxes > 0:
            chromosome[:, [0, 2]][xaxes_mask] = self.rand.integers(0, self.width, size = mutations_xaxes)
        
        if mutations_yaxes > 0:
            chromosome[:, [1, 3]][yaxes_mask] = self.rand.integers(0, self.height, size = mutations_yaxes)
        
        if mutations_rotangle > 0:
            chromosome[:, 4][rotangle_mask] = self.rand.integers(0, 360, size = mutations_rotangle)
        
        if mutations_color > 0:
            chromosome[:, 5:8][color_mask] = self.rand.integers(0, 255, size = mutations_color)

        if mutations_alpha > 0:
            chromosome[:, 8][mutations_alpha] = self.rand.uniform(size = mutations_alpha)

    def mutate_generation(self, fittest_threshold):
        mutation_matrix = self.rand.uniform(size = (self.population_size - fittest_threshold, 
                                            self.number_of_ellipses, 
                                            9))
        
        xaxes_mask = mutation_matrix[:, :, [0, 2]] < self.mutation_rate
        yaxes_mask = mutation_matrix[:, :, [1, 3]] < self.mutation_rate
        rotangle_mask = mutation_matrix[:, :, 4] < self.mutation_rate
        color_mask = mutation_matrix[:, :, 5:8] < self.mutation_rate
        alpha_mask = mutation_matrix[:, :, 8] < self.mutation_rate

        mutations_xaxes = np.sum(xaxes_mask)
        mutations_yaxes = np.sum(yaxes_mask)
        mutations_rotangle = np.sum(rotangle_mask)
        mutations_color = np.sum(color_mask)
        mutations_alpha = np.sum(alpha_mask)

        if mutations_xaxes > 0:
            self.current_generation[fittest_threshold:, :, [0, 2]][xaxes_mask] = self.rand.integers(0, self.width, size = mutations_xaxes)
        
        if mutations_yaxes > 0:
            self.current_generation[fittest_threshold:, :, [1, 3]][yaxes_mask] = self.rand.integers(0, self.height, size = mutations_yaxes)
        
        if mutations_rotangle > 0:
            self.current_generation[fittest_threshold:, :, 4][rotangle_mask] = self.rand.integers(0, 360, size = mutations_rotangle)
        
        if mutations_color > 0:
            self.current_generation[fittest_threshold:, :, 5:8][color_mask] = self.rand.integers(0, 255, size = mutations_color)

        if mutations_alpha > 0:
            self.current_generation[fittest_threshold:, :, 8][alpha_mask] = self.rand.uniform(size = mutations_alpha)
    
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
                    "number_of_ellipses": self.number_of_ellipses,
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

        res = PopulationEllipseRGBA(data["number_of_ellipses"], data["population_size"],
        data["width"], data["height"], data["expected_no_mutations"], data["rand_seed"])

        with open("./{}/last_generation.npy".format(root), "rb") as f:
            res.current_generation = np.load(f)
            for i in range(res.population_size):
                res.drawings[i] = res.embed_chromosome(res.current_generation[i])

        print("Finished loading the cached Population object.")
        return res