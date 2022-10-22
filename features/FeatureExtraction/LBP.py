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


if __name__ == '__main__':
    path = "C:\\Users\\tmaf\\Mega\\dataBase\\CLL"

    imgs = get_images(path)
    imagem = cv2.imread(imgs[0])
    b, g, r = cv2.split(imagem)
    hist1 = lbp(b, radius=2)
    hist2 = lbp(g, radius=2)
    hist3 = lbp(r, radius=2)

    plt.plot(hist1, color=(0, 0, 1, 0.4))
    plt.plot(hist2, color=(0, 1, 0, 0.4))
    plt.plot(hist3, color=(1, 0, 0, 0.4))

    path = "C:\\Users\\tmaf\\Mega\\dataBase\\FL"

    imgs = get_images(path)
    imagem = cv2.imread(imgs[0])
    b, g, r = cv2.split(imagem)
    hist1 = lbp(b, radius=2)
    hist2 = lbp(g, radius=2)
    hist3 = lbp(r, radius=2)

    plt.plot(hist1, color=(1, 0, 1, 0.4))
    plt.plot(hist2, color=(0, 1, 1, 0.4))
    plt.plot(hist3, color=(1, 1, 0, 0.4))

    plt.show()
