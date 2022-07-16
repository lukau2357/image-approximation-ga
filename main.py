import numpy as np
import cv2

from poly_rgb import PopulationPolyRBG
from poly_rgb_alpha import PopulationPolyRBGAlpha
from ga_engine import GAEvolver

if __name__ == "__main__":
    evolver = GAEvolver.load("mona_lisa_alpha")
    evolver.evolution()