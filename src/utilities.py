
def generateHistP(img):
    h = [0] * 256
    (rows, cols, culori) = img.shape
    for i in range(0, rows):
        for j in range(0, cols):
            h[img[i][j]] += 1.0
    for i in range(0, 256):
        h[i] /= rows * cols
    
    return h