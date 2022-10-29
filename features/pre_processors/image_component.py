import cv2


def image_component(image, channel):

    if channel == 'r':
        _, _, r = cv2.split(image)
        return r
    elif channel == 'g':
        _, g, _ = cv2.split(image)
        return g
    elif channel == 'b':
        b, _, _ = cv2.split(image)
        return b
    elif channel == 'h':
        h, _, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        return h
    elif channel == 's':
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, s, _ = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        return s
    elif channel == 'v':
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        return v
    elif channel == 'y':
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

