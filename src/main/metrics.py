import numpy as np
import math
import cv2

# the distance is computed based on color
from main.defs import Keypoint


def distanceImg(img, sablon, x_offset=0, y_offset=0, colorWeight=1):
    # assert (isinstance(img, cv2.))
    # assert (isinstance(sablon, cv2.img))
    (rows, cols, culori) = sablon.shape
    (rows2, cols2, culori2) = img.shape
    assert (rows <= rows2)
    assert (cols <= cols2)

    # print(sablon[0][0][0], img[0][0])
    # print(sablon[0][0].__class__)

    sum = 0
    for i in range(0, rows):
        for j in range(0, cols):
            sum += distance(sablon[i][j], img[y_offset + i][x_offset + j])

    return sum / (rows * cols)


# the distance is computed based on color and direction
# metric computed as weighted average
def distance(p1, p2):
    colorWeight = 1
    colorMetric = 0
    if isinstance(p1, list) or isinstance(p1, np.ndarray) and len(p1) == 3:
        colorMetric = np.sqrt(((int(p1[0]) - p2[0]) ** 2) +
                              ((int(p1[1]) - p2[1]) ** 2) +
                              ((int(p1[2]) - p2[2]) ** 2)) / np.sqrt((255 ** 2) * 3)
    else:
        if isinstance(p1, int):
            colorMetric = np.sqrt(int(p1 - p2) ** 2) / np.sqrt(255 ** 2)
        else:
            print("Bad input for distance metric")

    return colorMetric


# mutual information
def mutualInformation(img, sablon, x_offset=0, y_offset=0):
    # assert (isinstance(img, cv2.))
    # assert (isinstance(sablon, cv2.img))
    (rows, cols) = sablon.shape
    (rows2, cols2) = img.shape

    assert (rows <= rows2)
    assert (cols <= cols2)

    pA = [0] * 256
    pB = [0] * 256
    pAB = [None] * 256
    for i in range(256):
        pAB[i] = [0] * 256

    sum = 0
    for i in range(0, rows):
        for j in range(0, cols):
            pA[sablon[i][j]] += 1
            pB[img[y_offset + i][x_offset + j]] += 1
            pAB[sablon[i][j]][img[y_offset + i][x_offset + j]] += 1

    for i in range(0, 256):
        pA[i] /= rows * cols
        pB[i] /= rows * cols
        for j in range(0, 256):
            pAB[i][j] /= rows * cols

    for i in range(0, rows):
        for j in range(0, cols):
            currentPAB = pAB[sablon[i][j]][img[y_offset + i][x_offset + j]]
            numerator = pAB[sablon[i][j]][img[y_offset + i][x_offset + j]]
            denominator = pA[sablon[i][j]] * pB[img[y_offset + i][x_offset + j]]
            fraction = numerator / denominator
            if fraction < 1:
                fraction = 1
            sum += currentPAB * np.log(fraction)

    return sum


# generare de histograma probabilistica
def generateHistP(img):
    h = [0] * 256
    (rows, cols, culori) = img.shape
    for i in range(0, rows):
        for j in range(0, cols):
            h[img[i][j]] += 1.0
    for i in range(0, 256):
        h[i] /= rows * cols

    return h


def distanceK(keypointA, keypointB):
    colorWeight = 1
    directionWeight = 1

    assert (isinstance(keypointA, Keypoint))
    assert (isinstance(keypointB, Keypoint))

    colorMetric = np.sqrt(((keypointA.color[0] - keypointB.color[0]) ** 2) +
                          ((keypointA.color[1] - keypointB.color[1]) ** 2) +
                          ((keypointA.color[2] - keypointB.color[2]) ** 2))

    dimension = len(keypointA.direction)
    directionMetric = 0
    for i in range(dimension):
        directionMetric += (keypointA.direction[i] - keypointB.direction[i]) ** 2
    directionMetric = np.sqrt(directionMetric)

    return (directionWeight * directionMetric + colorWeight * colorMetric) / (colorWeight + directionWeight)


def cosineSimilarity(keypointA, keypointB):
    assert (isinstance(keypointA, Keypoint))
    assert (isinstance(keypointB, Keypoint))

    numerator = (keypointA.color[0] * keypointB.color[0] +
                 keypointA.color[1] * keypointB.color[1] +
                 keypointA.color[2] * keypointB.color[2])
    dimension = len(keypointA.direction)
    for i in range(dimension):
        numerator += keypointA.direction[i] * keypointB.direction[i]

    dimension = len(keypointA.color)
    denominatorA = 0
    denominatorB = 0
    for i in range(dimension):
        denominatorA += keypointA.color[i] ** 2
        denominatorB += keypointB.color[i] ** 2
    dimension = len(keypointA.direction)
    for i in range(dimension):
        denominatorA += keypointA.direction[i] ** 2
        denominatorB += keypointB.direction[i] ** 2

    return numerator / (np.sqrt(denominatorA) * np.sqrt(denominatorB))
