import numpy as np
from sklearn.decomposition import FastICA
import cv2
from matplotlib import pyplot as plt
import os


def get_images(path,type_file='.tif'):
    images = []
    for file in os.listdir(path):
        if file.endswith(type_file):
                images.append(str(os.path.join(path, file)))
    return images

def ica(image,components=10):
    """
    Analise de componentes independentes
    """
    ica = FastICA(n_components=components)
    ica.fit(image)
    return ica.fit_transform(image)
    



if __name__ == '__main__':

    path = "C:\\Users\\tmaf\\Mega\\dataBase\\CLL"
    
    imgs = get_images(path)
    imagem = cv2.imread(imgs[0])
    b,g,r = cv2.split(imagem)
    blue = ica(b)
    green = ica(g)
    red = ica(r)
    plt.subplot(131)
    plt.plot(blue,color=(0,0,1,0.8))
    plt.subplot(132)
    plt.plot(green,color=(0,1,0,0.8))
    plt.subplot(133)
    plt.plot(red,color=(1,0,0,0.8))
    plt.show()
    
