import numpy as np
from skimage import feature
import cv2
from matplotlib import pyplot as plt

import os


def get_images(path, type_file='.tif'):
    images = []
    for file in os.listdir(path):
        if file.endswith(type_file):
            images.append(str(os.path.join(path, file)))
    return images


def lbp(image, radius=2, n_points=8, method='uniform'):
    processed_image = feature.local_binary_pattern(image, n_points, radius, method)
    bins = int(processed_image.max() + 1)
    hist, _ = np.histogram(processed_image, density=False, bins=bins, range=(0, bins))
    return hist
