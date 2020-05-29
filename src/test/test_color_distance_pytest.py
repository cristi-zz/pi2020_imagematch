import cv2
import pytest
from main import metrics

IMAGE_FOLDER = '..\\..\\images\\'


def testColorDistanceOnPoints():
    gray1 = 0
    gray2 = 255
    metric = metrics.distance(gray1, gray2)
    assert metric > 0.9, "Testing on grayscale failed"
    color1 = [255, 120, 255]
    color2 = [0, 255, 0]
    metric = metrics.distance(color1, color2)
    assert metric > 0.8, "Testing on color1 failed"
    color1 = [255, 120, 255]
    color2 = [240, 100, 240]
    metric = metrics.distance(color1, color2)
    assert metric < 0.1, "Testing on color2 failed"


def testColorDistanceOnImageEasier():
    sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller2.png")
    test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png")
    assert sample is not None
    assert test is not None

    SIMILARITY_DISTANCE_THRESHOLD_EASIER = 0.05

    (rows, cols, colors1) = test.shape
    (rows2, cols2, colors2) = sample.shape

    matchOffsets = []

    for i in range(0, rows - rows2):
        for j in range(0, cols - cols2):
            dist = metrics.distanceImg(test, sample, j, i)
            if dist < SIMILARITY_DISTANCE_THRESHOLD_EASIER:
                matchOffsets.append((j, i))

    for (x, y) in matchOffsets:
        print("Match at offsets %d %d" % (x, y))

    assert len(matchOffsets) == 1


@pytest.mark.skip
def testColorDistanceOnImageMedium():
    sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png")
    test = cv2.imread(IMAGE_FOLDER + "colorFlareMedium.png")

    SIMILARITY_DISTANCE_THRESHOLD_MEDIUM = 0.20

    (rows, cols, colors1) = test.shape
    (rows2, cols2, colors2) = sample.shape

    matchOffsets = []

    for i in range(0, rows - rows2):
        for j in range(0, cols - cols2):
            dist = metrics.distanceImg(test, sample, j, i)
            if dist < SIMILARITY_DISTANCE_THRESHOLD_MEDIUM:
                matchOffsets.append((j, i, dist))

    for (x, y, d) in matchOffsets:
        print("Match at offsets %d %d with similarity %f" % (x, y, d))

    # assert len(matchOffsets) == 3
