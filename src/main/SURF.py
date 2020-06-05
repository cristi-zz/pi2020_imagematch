import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

from main import metrics

IMAGE_FOLDER = '..\\images\\'
secondaryImage = None
mainImage = None
greenToRedGradient = None
OUTPUT_DIRECTORY = "..\\..\\similarity_images"
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

def getImages():
    global secondaryImage
    global mainImage

    root = tk.Tk()
    rootDir = os.path.join(os.curdir, "..\\..\\images")
    root.withdraw()
    mainImage = cv2.imread("images\\colorFlareMedium.jpg", cv2.IMREAD_GRAYSCALE)
    secondaryImage = cv2.imread("images\\colorFlareSmall.jpg", cv2.IMREAD_GRAYSCALE)
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


    # cv2.waitKey()
    return


def getImage():
    global secondaryImage
    global mainImage

    root = tk.Tk()
    rootDir = os.path.join(os.curdir, "..\\..\\")
    root.withdraw()
    mainImage = cv2.imread("images\\colorFlareMedium.jpg", cv2.IMREAD_GRAYSCALE)
    secondaryImage = cv2.imread("images\\colorFlareSmall.jpg", cv2.IMREAD_GRAYSCALE)
    while mainImage is None:
        mainImagePath = filedialog.askopenfilename(initialdir=rootDir,
                                                   title="Select An Image",
                                                   )
        mainImage = cv2.imread(mainImagePath)
        if mainImage is None:
            messagebox.showerror("Error", "Bad image")


    # cv2.waitKey()
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

def incadreazaMask(imageMask, offset_x, offset_y, fill_x, fill_y, value):
    global gradientToValue

    if imageMask is None:
        print("Bad mask")
        return
    (rows, cols, color) = imageMask.shape

    if fill_y + offset_y > rows or fill_x + offset_x > cols:
        print("Bad fill")
        return

    for i in range(offset_y, fill_y + offset_y):
        # if gradientToValue[str(value)] > gradientToValue[imageMask[i][offset_x]]:
            imageMask[i][offset_x] = value
        # if gradientToValue[str(value)] > gradientToValue[imageMask[i][offset_x + fill_x- 1]]:
            imageMask[i][offset_x + fill_x- 1] = value


    for j in range(offset_x, fill_x + offset_x):
        # if gradientToValue[str(value)] > gradientToValue[imageMask[i][offset_x]]:
        #     imageMask[i][offset_x] = value
        # if gradientToValue[str(value)] > gradientToValue[imageMask[i][offset_x + fill_x - 1]]:
        #     imageMask[i][offset_x + fill_x - 1] = value
        imageMask[offset_y][j] = value
        imageMask[offset_y + fill_y - 1][j] = value
    return imageMask



def surf_similarity(similaritaty_percent = 75.00):

    (rows, cols, colors) = mainImage.shape
    (rows1, cols1, colors1) = secondaryImage.shape
    mask = np.zeros((rows, cols, 3), np.uint8)

    surf = cv2.xfeatures2d.SURF_create()


    kp1, des1 = surf.detectAndCompute(mainImage, None)
    kp2, des2 = surf.detectAndCompute(secondaryImage, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)


    # Apply ratio test
    similaritati = []
    potriviri = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            similaritati.append([m])
            a = len(similaritati)
            percent = (a * 100) / len(kp2)
            # print("{} % similarity".format(percent))
            if percent >= similaritaty_percent:
                potriviri.append([m])
                print('Match Found')
            else:
                print('Match not Found')


    img_final = cv2.drawMatchesKnn(mainImage, kp1, secondaryImage, kp2, similaritati, None, flags=2)
    plt.imshow(img_final), plt.show()

    cv2.imshow('Base image', mainImage)
    cv2.imshow('Patch image', secondaryImage)

    cv2.waitKey()

def surf_similarity_same_image(similaritaty_percent = 75.00):

    (rows, cols, colors) = mainImage.shape
    mask = np.zeros((rows, cols, 3), np.uint8)

    surf = cv2.xfeatures2d.SURF_create()


    kp1, des1 = surf.detectAndCompute(mainImage, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des1, k=2)

    nr_similaritati = 0
    nr_keypoinnturi = 0
    similaritati = []
    potriviri = []
    for m, n in matches:
        nr_keypoinnturi += 1
        if m.distance < 0.75 * n.distance:

            similaritati.append([m])
            a = len(similaritati)
            percent = (a * 100) / len(kp1)

            if m in potriviri:
                nr_similaritati += 1
            # print("{} % similarity".format(percent))
            if percent >= similaritaty_percent:
                potriviri.append([m])
            else:
                pass


    img_final = cv2.drawMatchesKnn(mainImage, kp1, mainImage, kp1, similaritati, None, flags=2)
    plt.imshow(img_final), plt.show()

    cv2.imshow('Base image', mainImage)

    cv2.waitKey()

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


def tearImage(mainImage):
    global greenToRedGradient
    buildGradient()
    # rupem imaginea intr-un grid de gridX x gridY
    (rows, cols, colors) = mainImage.shape
    similaritaty_percent = 75.00
    gridX = max(2, int(rows / 37))
    gridY = max(2, int(cols / 37))

    divX = int(rows / gridX)
    divY = int(cols / gridY)
    mask = np.zeros((rows, cols, 3), np.uint8)
    print()
    potriviri_mat = np.zeros((rows, cols), np.uint32)
    surf = cv2.xfeatures2d.SURF_create()
    print("Div: ", divX, divY)
    max_potriviri = 0
    for dim_x in range(0, rows - divX + 1, divX):
        for dim_y in range(0, cols - divY + 1, divY):
            print(dim_x, dim_y)
            auxImage = np.zeros((divX, divY, 3), np.uint8)
            for i in range(dim_x, dim_x + divX):
                for j in range(dim_y, dim_y + divY):
                    auxImage[i - dim_x][j - dim_y] = mainImage[i][j]
                    mainImage[i][j] = np.zeros(3)

            # cv2.imshow('Base image', mainImage)
            # cv2.imshow('Aux image', auxImage)
            # cv2.waitKey(1000)
            kp1, des1 = surf.detectAndCompute(mainImage, None)
            kp2, des2 = surf.detectAndCompute(auxImage, None)
            if (des2 is None):
                pass
            else:
                # BFMatcher with default params
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(des1, des2, k=2)

                # Apply ratio test
                similaritati = []
                potriviri = []
                nr_potriviri = 0
                try:
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            similaritati.append([m])
                            a = len(similaritati)
                            percent = (a * 100) / len(kp2)
                            # print("{} % similarity".format(percent))
                            if percent >= similaritaty_percent:
                                potriviri.append([m])
                                nr_potriviri += 1
                            else:
                                pass
                except:
                    pass
                # print(len(potriviri))
                potriviri_mat[dim_x][dim_y] = nr_potriviri
                max_potriviri = max(max_potriviri, nr_potriviri)
            for i in range(dim_x, dim_x + divX):
                for j in range(dim_y, dim_y + divY):
                    mainImage[i][j] = auxImage[i - dim_x][j - dim_y]


    for dim_x in range(0, rows - divX + 1, divX):
        for dim_y in range(0, cols - divY + 1, divY):
            print(dim_x, dim_y)
            print(greenToRedGradient[int(potriviri_mat[dim_x][dim_y] / max_potriviri * 100 - 0.1)])
            print(potriviri_mat[dim_x][dim_y])
            mainImage = incadreazaMask(imageMask=mainImage,
                            offset_x=dim_y,
                            offset_y=dim_x,
                            fill_x=divY,
                            fill_y=divX,
                            value=greenToRedGradient[int(potriviri_mat[dim_x][dim_y] / max_potriviri * 100 - 0.1)])
            cv2.imshow('Heatmap', mainImage)
            cv2.waitKey(500)

    # cv2.imshow('Base image', mainImage)
    cv2.imshow('Heatmap final', mainImage)
    cv2.imwrite(os.path.join(OUTPUT_DIRECTORY, "surf_petice.jpg"), mainImage)
    cv2.waitKey()

def main():
    # getImages()
    # surf_similarity()

    getImage()
    tearImage(mainImage)

    # surf_similarity_same_image()

if __name__ == "__main__":
    main()
