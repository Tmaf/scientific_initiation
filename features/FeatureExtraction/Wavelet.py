from pywt import dwt2


def wavelet(image, wavelet_name):
    approximation, (horizontal, vertical, diagonal) = dwt2(image, wavelet_name)
    return approximation, horizontal, vertical, diagonal
