import cv2


def dog(image, radius1, radius2):
    blur1 = cv2.GaussianBlur(image, (radius1, radius1), 0)
    blur2 = cv2.GaussianBlur(image, (radius2, radius2), 0)
    return blur2 - blur1
