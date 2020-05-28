import numpy as np
import math
import cv2

# the distance is computed based on color
from main.defs import Keypoint


def distanceImg(img, sablon, x_offset=0, y_offset=0, colorWeight=1):
    (rows, cols, culori) = sablon.shape
    (rows2, cols2, culori2) = img.shape
    assert (rows <= rows2)
    assert (cols <= cols2)

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
        colorA = np.array(p1, np.uint32)
        colorB = np.array(p2, np.uint32)
        colorMetric = np.sqrt(np.sum((colorA - colorB) * (colorA - colorB))) / np.sqrt((255**2)*3)
    else:
        if isinstance(p1, int):
            colorMetric = np.sqrt(int(p1 - p2) ** 2) / np.sqrt(255 ** 2)
        else:
            print("Bad input for distance metric")

    return colorMetric


# mutual information

epsilon = 1e-7

def mutualInformation(img, sablon, x_offset=0, y_offset=0):
    (rows, cols) = sablon.shape
    (rows2, cols2) = img.shape

    assert (rows <= rows2)
    assert (cols <= cols2)

    pA = np.zeros(256, dtype=np.float32)

    pB = np.zeros(256, dtype=np.float32)

    pAB = np.zeros((256, 256), dtype=np.float32)

    sum = 0
    for i in range(0, rows):
        for j in range(0, cols):
            pA[sablon[i][j]] += 1
            pB[img[y_offset + i][x_offset + j]] += 1
            pAB[sablon[i][j]][img[y_offset + i][x_offset + j]] += 1

    pA /= rows*cols
    pB /= rows*cols
    pAB /= rows*cols

    for i in range(0, rows):
        for j in range(0, cols):
            currentPAB = pAB[sablon[i][j]][img[y_offset + i][x_offset + j]]
            numerator = pAB[sablon[i][j]][img[y_offset + i][x_offset + j]]
            denominator = pA[sablon[i][j]] * pB[img[y_offset + i][x_offset + j]]
            fraction = numerator / (denominator + epsilon)
            sum += currentPAB * np.log(1+fraction)

    return sum


# generare de histograma probabilistica
def generateHistP(img):
    h = [0] * 256
    (rows, cols, colors) = img.shape
    for i in range(0, rows):
        for j in range(0, cols):
            h[img[i][j]] += 1.0
    for i in range(0, 256):
        h[i] /= rows * cols

    return h


def distanceK(keypointA, keypointB):
    colorWeight = 1
    directionWeight = 0

    assert (isinstance(keypointA, Keypoint))
    assert (isinstance(keypointB, Keypoint))

    colorA = np.array(keypointA.color, np.uint32)
    colorB = np.array(keypointB.color, np.uint32)
    directionA = np.array(keypointA.direction, np.uint32)
    directionB = np.array(keypointB.direction, np.uint32)

    colorMetric = np.sqrt(np.sum((colorA - colorB) * (colorA - colorB)))

    directionMetric = np.sqrt(np.sum((directionA - directionB) * (directionA - directionB)))

    return (directionWeight * directionMetric + colorWeight * colorMetric) / (colorWeight + directionWeight)


def cosineSimilarityImg(img, sablon, x_offset=0, y_offset=0):
    (rows, cols, colors) = sablon.shape
    (rows2, cols2, colors2) = img.shape
    assert (rows <= rows2)
    assert (cols <= cols2)

    sum1 = 0
    for i in range(0, rows):
        for j in range(0, cols):
            keypointA = Keypoint([i, j],
                                 sablon[i][j],
                                 [1, 1, 1])
            keypointB = Keypoint([y_offset + i, x_offset + j],
                                 img[y_offset + i, x_offset + j],
                                 [1, 1, 1])
            sum1 += cosineSimilarity(keypointA, keypointB)

    return sum1 / (rows * cols)


def cosineSimilarity(keypointA, keypointB):
    assert (isinstance(keypointA, Keypoint))
    assert (isinstance(keypointB, Keypoint))
    colorWeight = 1
    directionWeight = 0

    colorA = np.array(keypointA.color, np.uint32)
    colorB = np.array(keypointB.color, np.uint32)
    directionA = np.array(keypointA.direction, np.uint32)
    directionB = np.array(keypointB.direction, np.uint32)

    keypointA.color = colorA
    keypointB.color = colorB
    keypointA.direction = directionA
    keypointB.direction = directionB

    numerator1 = np.dot(colorA, colorB) * colorWeight
    numerator2 = np.dot(directionA, directionB) * directionWeight
    numerator = (numerator1 + numerator2) / (colorWeight + directionWeight)

    denominatorA = np.dot(colorA, colorA) * colorWeight
    denominatorA2 = np.dot(directionA, directionA) * directionWeight
    denominatorA = (denominatorA + denominatorA2) / (colorWeight + directionWeight)

    denominatorB = np.dot(colorB, colorB) * colorWeight
    denominatorB2 = np.dot(directionB, directionB) * directionWeight
    denominatorB = (denominatorB + denominatorB2) / (colorWeight + directionWeight)

    return numerator / (np.sqrt(denominatorA) * np.sqrt(denominatorB))
