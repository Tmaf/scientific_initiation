import os
import numpy as np
from skimage import feature,exposure
import cv2
from matplotlib import pyplot as plt


def get_images(path, type_file='.tif'):
    images = []
    for file in os.listdir(path):
        if file.endswith(type_file):
                images.append(str(os.path.join(path, file)))
    return images


def hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1)):
    """
    Histograma de Gradientes Orientados.
    """
    fd, hog_image = feature.hog(image, orientations, pixels_per_cell,cells_per_block,block_norm='L2-Hys',visualize=True, multichannel=False)
    return (fd,hog_image)

if __name__ == '__main__':

    path = "C:\\Users\\tmaf\\Mega\\dataBase\\CLL"
    
    imgs = get_images(path)
    imagem = cv2.imread(imgs[0])
    b,g,r = cv2.split(imagem)
    f,h=hog(b,pixels_per_cell=(16,16),cells_per_block=(1, 1),orientations=8)
    f2,h2=hog(g,pixels_per_cell=(16,16),cells_per_block=(1, 1),orientations=8)
    f3,h3=hog(r,pixels_per_cell=(16,16),cells_per_block=(1, 1),orientations=8)
    hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 10))
    cv2.imshow('teste',hog_image_rescaled)
    plt.plot(f,color=(1,0,0,0.4))
    plt.plot(f2,color=(0,1,0,0.4))
    plt.plot(f3,color=(0,0,1,0.4))
    plt.show()
