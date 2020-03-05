import cv2
import numpy as np

if __name__ == "__main__":
    print("Hello!")
    img = np.zeros((128,128, 3), dtype=np.uint8)
    img[:,:,1] = 128
    cv2.imshow("img", img)
    cv2.waitKey()