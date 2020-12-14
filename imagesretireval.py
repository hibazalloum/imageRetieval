import cv2 as cv
import numpy as np
import glob
import pandas as pd
import math

img_format = ['jpg', 'png', 'gif', 'jpeg']
h = 0
for frmt in img_format:
    for img_name in glob.glob('*.' + frmt):
        filename = img_name[img_name.rfind("/") + 1:]
        img = cv.imread(img_name)

        listcolor = []
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv.calcHist([img], [i], None, [256], [0, 256])  # calculate histogram for blue green red color
            histnormalize1 = cv.normalize(hist, hist)  # calculate Normalize for histograms
            listcolor.append(histnormalize1)
            # print(listcolor)
            # print(histnormalize1)
            # print(len(histnormalize1))

        df = pd.DataFrame({'name': img_name, 'histogram': [listcolor]})

        if h == 0:
            df.to_pickle('my_csv.pickle')
            h = 1
        else:
            df.to_pickle('my_csv.pickle')


image_query = cv.imread('images (6).jpeg')
listcolor = []
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv.calcHist([image_query], [i], None, [256], [0, 256])  # calculate histogram for blue green red color
    histnormalize1 = cv.normalize(hist, hist)  # calculate Normalize for histograms
    listcolor.append(histnormalize1)




histogram_pickle = pd.read_pickle('my_csv.pickle')
