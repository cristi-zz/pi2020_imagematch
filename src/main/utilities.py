import numpy as np

# generare de histograma probabilistica
def generateHistP(img):
    h = np.zeros(256, dtype=np.float32)

    (rows, cols, colors) = img.shape
    for i in range(0, rows):
        for j in range(0, cols):
            h[img[i][j]] += 1.0

    h /= rows * cols

    return h