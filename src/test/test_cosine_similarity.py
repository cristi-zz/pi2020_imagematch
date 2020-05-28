import unittest
import cv2
from main.defs import Keypoint
from main import metrics

IMAGE_FOLDER = '..\\..\\images\\'


class TestCosineSimilarity(unittest.TestCase):
    def testCosineSimilarityOnPoints(self):
        keypointA = Keypoint([0, 0],
                             [120, 120, 120],
                             [0, 3, 1])
        keypointB = Keypoint([50, 50],
                             [120, 120, 120],
                             [0, 3, 1])
        metric = metrics.cosineSimilarity(keypointA, keypointB)
        self.assertGreater(metric, 0.99, "Testing on color 1")
        keypointA = Keypoint([0, 0],
                             [120, 120, 120],
                             [0, 3, 1])
        keypointB = Keypoint([50, 50],
                             [255, 255, 255],
                             [0, 9, 3])
        metric = metrics.cosineSimilarity(keypointA, keypointB)
        self.assertGreater(metric, 0.99, "Testing on color 2")
        keypointA1 = Keypoint([0, 0],
                              [255, 255, 255],
                              [0, 3, 1])
        keypointB1 = Keypoint([50, 50],
                              [164, 73, 163],
                              [12, 12, 1])
        metric = metrics.cosineSimilarity(keypointA1, keypointB1)


    #@unittest.skip
    def testCosineSimilarityOnImageEasier(self):
        sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller.png")
        test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png")

        SIMILARITY_DISTANCE_THRESHOLD_EASIER = 0.90

        (rows, cols, colors1) = test.shape
        (rows2, cols2, colors2) = sample.shape

        matchOffsets = []

        for i in range(0, rows - rows2):
            for j in range(0, cols - cols2):
                dist = metrics.cosineSimilarityImg(test, sample, j, i)
                if dist >= SIMILARITY_DISTANCE_THRESHOLD_EASIER:
                    matchOffsets.append((j, i))

        for (x, y) in matchOffsets:
            print("Match at offsets %d %d" % (x, y))

        self.assertEqual(len(matchOffsets), 1)

    @unittest.skip
    def testColorDistanceOnImageSmall2(self):
        sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller2.png")
        test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png")

        SIMILARITY_DISTANCE_THRESHOLD_EASY2 = 0.9999

        (rows, cols, colors1) = test.shape
        (rows2, cols2, colors2) = sample.shape

        matchOffsets = []

        for i in range(0, rows - rows2):
            for j in range(0, cols - cols2):
                dist = metrics.cosineSimilarityImg(test, sample, j, i)
                if dist >= SIMILARITY_DISTANCE_THRESHOLD_EASY2:
                    matchOffsets.append((j, i, dist))

        for (x, y, d) in matchOffsets:
            print("Match at offsets %d %d with similarity %f" % (x, y, d))

        self.assertEqual(len(matchOffsets), 1)


if __name__ == '__main__':
    unittest.main()
