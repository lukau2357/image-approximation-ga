import numpy as np
import cv2

from poly_rgb_alpha import PopulationPolyRBGAlpha
from ellipse_rgb_alpha import PopulationEllipseRGBA
from ga_engine import GAEvolver

def image_evolution_ellipses(root, ext):
    target = cv2.imread("./images/{}.{}".format(root, ext))
    population = PopulationEllipseRGBA(50, 200, target.shape[1], target.shape[0], 4, 41)
    evolver = GAEvolver("lets_see", target, "./images/{}.{}".format(root, ext), 3000, 0.5, 
                        population, "psnr", "cuts", True)
    evolver.evolution()

def image_evolution_poly(root, ext):
    target = cv2.imread("./images/{}.{}".format(root, ext))
    population = PopulationPolyRBGAlpha(3, 200, 200, target.shape[1], target.shape[0], 4, 41)
    evolver = GAEvolver("lets_see", target, "./images/{}.{}".format(root, ext), 3000, 0.5, 
                        population, "psnr", "cuts", True)
    evolver.evolution()

def load_evolver(root):
    evolver = GAEvolver.load(root)
    return evolver

def render_best(evolver : GAEvolver):
    path = "./{}/best_chromosomes_history/{}.npy".format(evolver.alg_label, evolver.last_generation_index - 1)
    chromosome = np.load(path)
    image = evolver.population.embed_chromosome(chromosome)
    cv2.imshow("{}-{}".format(evolver.alg_label, evolver.last_generation_index - 1), image)
    cv2.waitKey(0)

if __name__ == "__main__":
    target = cv2.imread("./images/mona_lisa.jpg")
    population = PopulationPolyRBGAlpha(6, 150, 200, target.shape[1], target.shape[0], 8, 12)
    evolver = GAEvolver("mona_lisa_alpha_v2", target, "./images/mona_lisa.jpg", 15000, 0.5, population, "psnr", "uniform", caching = True, caching_period = 1000)
    evolver.evolution()