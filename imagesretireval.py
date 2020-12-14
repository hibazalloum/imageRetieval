import cv2 as cv
import numpy as np
import glob
import pandas as pd
import math

img_format = ['jpg', 'png', 'gif', 'jpeg']
h = 0

for frmt in img_format:
    for img_name in glob.glob('*.' + frmt):
        img = cv.imread(img_name)
        hist_b = cv.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv.calcHist([img], [1], None, [256], [0, 256])
        hist_r = cv.calcHist([img], [2], None, [256], [0, 256])


        hist_bn = np.zeros(256)
        hist_gn = np.zeros(256)
        hist_rn = np.zeros(256)


        for i in range(256):
            hist_bn[i] = hist_b[i][0]
            hist_gn[i] = hist_g[i][0]
            hist_rn[i] = hist_r[i][0]

        histnormalizeB = cv.normalize(hist_bn, hist_b, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        histnormalizeG = cv.normalize(hist_gn, hist_g, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
        histnormalizeR = cv.normalize(hist_rn, hist_r, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

        df = pd.DataFrame({'name': img_name,
                           'Blue': [histnormalizeB],
                           'Green': [histnormalizeG],
                           'Red': [histnormalizeR]})


        if h == 0:
            df.to_csv('my_csv.csv', index=False)
            h = 1
        else:
            df.to_csv('my_csv.csv', mode='a', header=False, index=False)

#histogram_csv = pd.read_csv('my_csv.csv')
#histogram_df = pd.DataFrame(histogram_csv)
#histogram_compact = histogram_df[['img_name'], ['blue'], ['green'], ['red']]

# Euclidean Distance
#hist_dist = sqrt( sum( ( a[i][j] - b[i][j] )^2 ) ) for all i = 0..width, j = 0..height
image_query = cv.imread('images (6).jpeg')
hist_b = cv.calcHist([image_query], [0], None, [256], [0, 256])
hist_g = cv.calcHist([image_query], [1], None, [256], [0, 256])
hist_r = cv.calcHist([image_query], [2], None, [256], [0, 256])


hist_bn = np.zeros(256)
hist_gn = np.zeros(256)
hist_rn = np.zeros(256)


for i in range(256):
    hist_bn[i] = hist_b[i][0]
    hist_gn[i] = hist_g[i][0]
    hist_rn[i] = hist_r[i][0]

    querynormalizeB = cv.normalize(hist_bn, hist_b, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    querynormalizeG = cv.normalize(hist_gn, hist_g, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    querynormalizeR = cv.normalize(hist_rn, hist_r, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

#print(histnormalizeR)
#print(histnormalizeB)
#print(histnormalizeG)

histogram_csv = pd.read_csv('my_csv.csv', delimiter=',', sep=)
histogram_df = pd.DataFrame(histogram_csv)
bgr = histogram_df[['Blue', 'Green', 'Red']]
blue = bgr['Blue']

print(blue)
# Euclidean Distance
#for i in range(blue):
    #print(i)
