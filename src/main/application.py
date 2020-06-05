import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

import numpy as np
import cv2
import os

from main import metrics

secondaryImage = None
mainImage = None
greenToRedGradient = None
gradientToValue = None

OUTPUT_DIRECTORY = "..\\..\\similarity_images\\"


def getImages():
    global secondaryImage
    global mainImage

    root = tk.Tk()
    rootDir = os.path.join(os.curdir, "..\\..\\")
    root.withdraw()
    mainImage = None  # cv2.imread("..\\..\\images\\colorFlareSmall.png")
    secondaryImage = None  # cv2.imread("..\\..\\images\\colorFlareSmaller.png")
    while mainImage is None:
        mainImagePath = filedialog.askopenfilename(initialdir=rootDir,
                                                   title="Select An Image",
                                                   )
        mainImage = cv2.imread(mainImagePath)
        if mainImage is None:
            messagebox.showerror("Error", "Bad image")

    while secondaryImage is None:
        secondImagePath = filedialog.askopenfilename(initialdir=rootDir,
                                                     title="Select A Patch For Comparison",
                                                     )
        secondaryImage = cv2.imread(secondImagePath)
        if secondaryImage is None:
            messagebox.showerror("Error", "Bad image")
    cv2.imshow('Base image', mainImage)
    cv2.imshow('Patch image', secondaryImage)
    return


def fillMask(imageMask, offset_x, offset_y, fill_x, fill_y, value):
    global gradientToValue

    if imageMask is None:
        print("Bad mask")
        return
    (rows, cols, color) = imageMask.shape

    if fill_y + offset_y > rows or fill_x + offset_x > cols:
        print("Bad fill")
        return

    for i in range(offset_y, fill_y + offset_y):
        for j in range(offset_x, fill_x + offset_x):
            currentColor = imageMask[i][j]
            if gradientToValue[str(value)] > gradientToValue[str(currentColor)]:
                imageMask[i][j] = value

    return imageMask


def buildGradient():
    # BGR
    # green to yellow
    global greenToRedGradient
    global gradientToValue

    gradientToValue = {}
    greenToRedGradient = np.zeros((101, 3), np.uint8)
    for i in range(40):
        greenToRedGradient[i] = np.zeros(3, np.uint8)
        greenToRedGradient[i][1] = 255
        greenToRedGradient[i][2] = 255 * i / 60
    # yellow to orange
    for i in range(40, 60):
        greenToRedGradient[i] = np.zeros(3, np.uint8)
        greenToRedGradient[i][1] = 255
        greenToRedGradient[i][2] = 255 * (i / 60)
    for i in range(60, 80):
        greenToRedGradient[i] = np.zeros(3, np.uint8)
        greenToRedGradient[i][1] = (255 * (80 - i / 80)) + 15
        greenToRedGradient[i][2] = greenToRedGradient[59][2]
    # orange to red
    for i in range(80, 100):
        greenToRedGradient[i] = np.zeros(3, np.uint8)
        greenToRedGradient[i][1] = greenToRedGradient[79][1] * ((100 - ((i - 80) * 5)) / 100)
        greenToRedGradient[i][2] = greenToRedGradient[59][2]
    for i in range(101):
        gradientToValue[str(greenToRedGradient[i])] = i
    zero = np.array([0, 0, 0], np.uint8)
    gradientToValue[str(zero)] = -1


def colorDistance(mainImage, secondaryImage):
    global greenToRedGradient

    (rows, cols, colors) = mainImage.shape
    (rows1, cols1, colors1) = secondaryImage.shape
    mask = np.zeros((rows, cols, 3), np.uint8)
    for i in range(rows):
        for j in range(cols):
            mask[i][j] = greenToRedGradient[0]

    for i in range(0, rows - rows1):
        for j in range(0, cols - cols1):
            dist = metrics.distanceImg(mainImage, secondaryImage, j, i)
            # print(i, j)
            mask = fillMask(imageMask=mask,
                            offset_x=j,
                            offset_y=i,
                            fill_x=cols1,
                            fill_y=rows1,
                            value=greenToRedGradient[99 - int(dist * 100)])

    return mask
    # cv2.imshow("Mask", mask)
    # cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "distance_medium.jpg"), mask)
    # #cv2.waitKey()


def mutualInformation(mainImage, secondaryImage):
    global greenToRedGradient

    mainImage = cv2.cvtColor(mainImage, cv2.COLOR_BGR2GRAY)
    secondaryImage = cv2.cvtColor(secondaryImage, cv2.COLOR_BGR2GRAY)
    (rows, cols) = mainImage.shape
    (rows1, cols1) = secondaryImage.shape
    mask = np.zeros((rows, cols, 3), np.uint8)
    for i in range(rows):
        for j in range(cols):
            mask[i][j] = greenToRedGradient[0]

    for i in range(0, rows - rows1):
        for j in range(0, cols - cols1):
            dist = metrics.mutualInformation(mainImage, secondaryImage, j, i)
            print(dist)
            # print(i, j)
            mask = fillMask(imageMask=mask,
                            offset_x=j,
                            offset_y=i,
                            fill_x=cols1,
                            fill_y=rows1,
                            value=greenToRedGradient[100 - int(dist * 10)])

    return mask
    # cv2.imshow("Mask", mask)
    # cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "distance_medium.jpg"), mask)
    # cv2.waitKey()


def cosineDistance(mainImage, secondaryImage):
    global greenToRedGradient

    (rows, cols, colors) = mainImage.shape
    (rows1, cols1, colors1) = secondaryImage.shape
    mask = np.zeros((rows, cols, 3), np.uint8)
    for i in range(rows):
        for j in range(cols):
            mask[i][j] = greenToRedGradient[0]

    for i in range(0, rows - rows1):
        for j in range(0, cols - cols1):
            dist = metrics.cosineSimilarityImg(mainImage, secondaryImage, j, i)
            # print(i, j)
            mask = fillMask(imageMask=mask,
                            offset_x=j,
                            offset_y=i,
                            fill_x=cols1,
                            fill_y=rows1,
                            value=greenToRedGradient[int(dist * 100)])

    return mask
    # cv2.imshow("Mask", mask)
    # cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "cosine_medium.jpg"), mask)
    # #cv2.waitKey()


def showoffGradient():
    global greenToRedGradient

    gradientImg = np.zeros((100, 500, 3), np.uint8)
    for i in range(500):
        for j in range(100):
            gradientImg[j][i] = greenToRedGradient[int(i / 5)]
    cv2.imshow("Gradient", gradientImg)
    cv2.waitKey()


def tearImage(mainImage):
    # rupem imaginea intr-un grid de 4x6
    (rows, cols, colors) = mainImage.shape

    gridX = 4
    gridY = 6
    auxImage = None
    divX = int(rows / gridX)
    divY = int(cols / gridY)
    maskCount = 0
    for dim_x in range(0, rows - divX + 1, divX):
        for dim_y in range(0, cols - divY + 1, divY):
            auxImage = np.zeros((divX, divY, 3), np.uint8)
            for i in range(dim_x, dim_x + divX):
                for j in range(dim_y, dim_y + divY):
                    auxImage[i - dim_x][j - dim_y] = mainImage[i][j]
                    mainImage[i][j] = np.zeros(3)

            # prelucrari aux
            # mask = cosineDistance(mainImage, auxImage)
            # mask = np.concatenate((mask, mainImage), axis=1)
            # #mask = np.concatenate((mainImage, mask), axis=1)
            # cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "cosine_big_" + str(maskCount) + ".jpg"), mask)
            #
            # maskCount += 1
            # print(maskCount)
            cv2.imshow("aux",auxImage)
            cv2.imshow("main",mainImage)
            cv2.waitKey()
            for i in range(dim_x, dim_x + divX):
                for j in range(dim_y, dim_y + divY):
                    mainImage[i][j] = auxImage[i - dim_x][j - dim_y]


def main():
    global mainImage, secondaryImage
    buildGradient()
    # getImages()
    mainImage = cv2.imread("..\\..\\images\\colorFlareSmall.png")
    tearImage(mainImage)
    # colorDistance()
    # cosineDistance()
    # mutualInformation(mainImage, secondaryImage)


if __name__ == "__main__":
    main()
