import cv2 as cv
import numpy as np
import glob
import pandas as pd


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

        df = pd.DataFrame({'name': img_name,
                            'Blue': hist_bn,
                            'Green': hist_gn,
                            'Red': hist_rn})
        if h == 0:
            df.to_csv('my_csv.csv', index=False)
            h = 1
        else:
            df.to_csv('my_csv.csv', mode='a', header=False, index=False)
