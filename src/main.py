import cv2
import numpy as np
import os
import metrics
IMAGE_FOLDER = '..\\.images\\'

def test_distanta_culori():
    sample = cv2.imread(IMAGE_FOLDER + "peisaj_fragment.png")
    test = cv2.imread(IMAGE_FOLDER + "peisaj.png")

    (rows, cols, culori2) = test.shape
    (rows2, cols2, culori2) = sample.shape   
    min1 = 0x3f3f3f3f3f
    #print(metrics.distance_img(test, sample, j, i))

    for i in range(0, rows - rows2):
        for j in range(0, cols - cols2):
            dist = metrics.distance_img(test, sample, j, i)
            if min1 > dist:
                min1 = dist
                print(i, j, dist)

    
    cv2.imshow("test", test)
    cv2.imshow("sample", sample)
    cv2.waitKey()

def test_mutual_information():
    sample = cv2.imread(IMAGE_FOLDER + "peisaj_fragment.png", cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(IMAGE_FOLDER + "peisaj_fragment2.png", cv2.IMREAD_GRAYSCALE)

    (rows, cols) = test.shape
    (rows2, cols2) = sample.shape   
    min1 = 0x3f3f3f3f3f
    #print(metrics.distance_img(test, sample, j, i))

    dist = metrics.mutual_information(test, sample)
    print(dist)
    cv2.imshow("test", test)
    cv2.imshow("sample", sample)
    cv2.waitKey()


if __name__ == "__main__":
    test_mutual_information()