import unittest
import cv2

from main import metrics

IMAGE_FOLDER = '..\\..\\images\\'


class TestMutualInformation(unittest.TestCase):
    def testMutualInformationSmall(self):
        sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller.png", cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png", cv2.IMREAD_GRAYSCALE)

        (rows, cols) = test.shape
        (rows2, cols2) = sample.shape
        maxInformation = 0
        maxCoord = ()

        for i in range(0, rows - rows2):
            for j in range(0, cols - cols2):
                mutualInformation = metrics.mutualInformation(test, sample, j, i)
                if mutualInformation > maxInformation:
                    maxInformation = mutualInformation
                    maxCoord = (j, i)

        self.assertEqual(maxCoord[0], 34)
        self.assertEqual(maxCoord[1], 45)

    def testMutualInformationSmall2(self):
        sample = cv2.imread(IMAGE_FOLDER + "colorFlareSmaller2.png", cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(IMAGE_FOLDER + "colorFlareSmall.png", cv2.IMREAD_GRAYSCALE)

        (rows, cols) = test.shape
        (rows2, cols2) = sample.shape
        maxInformation = 0
        maxCoord = ()

        for i in range(0, rows - rows2):
            for j in range(0, cols - cols2):
                mutualInformation = metrics.mutualInformation(test, sample, j, i)
                if mutualInformation > maxInformation:
                    maxInformation = mutualInformation
                    maxCoord = (j, i)

        self.assertEqual(maxCoord[0], 0)
        self.assertEqual(maxCoord[1], 0)


if __name__ == '__main__':
    unittest.main()
