import unittest
import cv2
from main import metrics

IMAGE_FOLDER = '..\\..\\images\\'


class TestColorDistance(unittest.TestCase):
    def testColorDistanceOnPoints(self):
        gray1 = 0
        gray2 = 255
        metric = metrics.distance(gray1, gray2)
        self.assertGreater(metric, 0.9, "Testing on grayscale")
        color1 = [255, 120, 255]
        color2 = [0, 255, 0]
        metric = metrics.distance(color1, color2)
        self.assertGreater(metric, 0.8, "Testing on color 1")
        color1 = [255, 120, 255]
        color2 = [240, 100, 240]
        metric = metrics.distance(color1, color2)
        self.assertGreater(0.1, metric, "Testing on color 2")

    # @unittest.skip
    def testColorDistanceOnImageEasier(self):
        sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller.png")
        test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png")

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

        self.assertEqual(len(matchOffsets), 1)

    @unittest.skip
    def testColorDistanceOnImageMedium(self):
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

        #self.assertEqual(len(matchOffsets), 3)


if __name__ == '__main__':
    unittest.main()
