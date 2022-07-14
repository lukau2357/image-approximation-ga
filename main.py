import numpy as np
import cv2

from population_poly import PopulationPolyRBG
from ga_engine import GAEvolver

if __name__ == "__main__":
    # target = cv2.imread("./images/mona_lisa_2.jpg")
    # test_obj = PopulationPolyRBG(6, 150, 200, target.shape[1], target.shape[0], 2, 41)
    # evolver = GAEvolver("mona_lisa_test", target, "images/mona_lisa_2.jpg",
    #                    20, 0.5, test_obj, "psnr", "cuts", True, caching = True)

    # evolver.evolution()

    evolver = GAEvolver.load("mona_lisa_test")
    evolver.evolution()