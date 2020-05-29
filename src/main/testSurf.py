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


def getImages():
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

    while secondaryImage is None:
        secondImagePath = filedialog.askopenfilename(initialdir=rootDir,
                                                     title="Select A Patch For Comparison",
                                                     )
        secondaryImage = cv2.imread(secondImagePath)
        if secondaryImage is None:
            messagebox.showerror("Error", "Bad image")


    # cv2.waitKey()
    return


def fillMask(imageMask, offset_x, offset_y, fill_x, fill_y, value):
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
            if value > currentColor:
                imageMask[i][j] = value



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
                print('Match Found')
            else:
                print('Match not Found')


    img_final = cv2.drawMatchesKnn(mainImage, kp1, mainImage, kp1, similaritati, None, flags=2)
    plt.imshow(img_final), plt.show()

    cv2.imshow('Base image', mainImage)

    cv2.waitKey()


def main():
    getImages()
    surf_similarity()
    # surf_similarity_same_image()

if __name__ == "__main__":
    main()
