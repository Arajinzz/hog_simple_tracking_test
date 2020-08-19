import numpy as np
import cv2

def normSSD(b1, b2):
    return ((b1-b2)**2).sum() / ((b1**2).sum()**0.5 * (b2**2).sum()**0.5)

def getAngles(img):
    # gradient
    Gx, Gy = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1), cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

    angles = (np.arctan2(Gy, Gx) * 180.0) / np.pi
    angles[angles < 0] += 360

    return angles


def histogram(angles, bins, bin_size):
    angs = np.zeros(bins)
    inds = np.uint(angles // bin_size)
    np.add.at(angs, inds.flatten(), 1)
    return angs


def HOG(img, bins, bin_size):
    angs = getAngles(img)
    return histogram(angs, bins, bin_size)
    