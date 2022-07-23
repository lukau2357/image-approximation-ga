import numpy as np
import cv2
import time

from sklearn.cluster import DBSCAN

def kmeans_color_palette(image : np.ndarray, clusters : int, iterations : int,\
                         repeats : int):
    '''
    Performs the K-means clustering algorithm on the given image and returns the color 
    palette for the input image.
    '''
    input = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, iterations, -1)
    _, __, centers = cv2.kmeans(input, clusters, None, criteria, repeats, cv2.KMEANS_RANDOM_CENTERS)
    centers = centers.astype(np.int32)
    return centers

def dbscan_color_palette(image : np.ndarray, 
                        eps : float = 0.5, 
                        min_samples : int = 5, 
                        metric : str = "euclidean"):
    '''
    Performs the DBSCAN clustering algorithm on the given RGB image and returns the color 
    palette for the input image. Colors for individual clusters are computed as cluster-wise
    means.
    '''

    height = image.shape[0]
    width = image.shape[1]

    dbscan_input = np.empty((height * width, 3))
    dbscan_input[:, 0] = image[:, :, 0].flatten().copy()
    dbscan_input[:, 1] = image[:, :, 1].flatten().copy()
    dbscan_input[:, 2] = image[:, :, 2].flatten().copy()

    dbscan = DBSCAN(eps = eps, min_samples = min_samples, metric = metric)
    labels = dbscan.fit_predict(dbscan_input)
    return labels

def rand_color_palette(rand : np.random.Generator, num_colors : int):
    return rand.integers(0, 256, size = (num_colors, 3))
