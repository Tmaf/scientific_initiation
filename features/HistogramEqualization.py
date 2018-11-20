import cv2


def histogram_equalization(image):
    return cv2.equalizeHist(image)
