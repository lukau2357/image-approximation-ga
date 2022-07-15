import numpy as np
import cv2

from poly_rgb import PopulationPolyRBG
from poly_rgb_alpha import PopulationPolyRBGAlpha
from ga_engine import GAEvolver

if __name__ == "__main__":
    target = cv2.imread("./images/moon.jpeg")
    generations = 10

    test_obj = PopulationPolyRBGAlpha(6, 150, 200, target.shape[1], target.shape[0], 2, 41)
    evolver = GAEvolver("moon_alpha", target, "images/moon.jpeg",
                       generations, 0.5, test_obj, "psnr", "uniform", True, caching = True)

    evolver.evolution()
    evolver.render(generations - 1) 