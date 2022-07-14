import numpy as np
import cv2

def kmeans_color_palette(image : np.ndarray, clusters : int, iterations : int,\
                         repeats : int):
    input = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, iterations, -1)
    c, l, centers = cv2.kmeans(input, clusters, None, criteria, repeats, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.int32)
    return centers