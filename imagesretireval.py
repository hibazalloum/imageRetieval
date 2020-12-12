import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

class Images:
    def __init__(self, image, path):
        self.path = path
        self.image = image

    def reSize(self):
        img = self.image
        img1 = cv.imread(img)
        reSize = cv.resize(img1, (1400 * 1400))
        return reSize

    def RGBHistogram(self):
        image_hight = self.image.shape[0]
        image_width = self.image.shape[1]
        image_channels = self.image.shape[2]
        hist = np.zeros([256,image_channels], np.int)
        for x in range(0, image_hight):
            for y in range(0, image_width):
                for c in range(0, image_channels):
                    hist[self.image[x,y,c]] += 1
        for i in range(0, hist.shape[0]):
            for c in range(0,hist.shape[1]):
                return i,c, hist[i,c]


    def histSaving(self):
        path = self.path
        for file in os.listdir(path):
            if os.path.isfile(path + file):
                image = cv.imread(img)

'''

def histogram(path):
    # path = path
    img = image
    for infile in os.listdir(path):
        if os.path.isfile(path + infile):
            image = cv.imread(img)
            file, ext = os.path.split(path + infile)
            image.reshap(1400, 1400)
            hist = cv.calcHist([image], [0], None, [256], [0, 256])
            image.save(file + 'reshape.jpg' + 'JPEG')
            data = np.loadtxt('data.txt')
'''


def RGBHistogram(image):
    img = cv.imread(image)
    image_hight = img.shape[0]
    image_width = img.shape[1]
    image_channels = img.shape[2]
    hist = np.zeros([256, image_channels], np.int)
    for x in range(0, image_hight):
        for y in range(0, image_width):
            for c in range(0, image_channels):
                hist[img[x, y, c]] += 1
            for i in range(0, hist.shape[0]):
                for c in range(0, hist.shape[1]):
                    return i, c, hist[i, c]



if __name__ == '__main__':
    img = cv.imread('baby.jpg')
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    print(hist)

