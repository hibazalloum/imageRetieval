import cv2 as cv
import numpy as np
import glob
import pandas as pd

def histograms(directory, name):
    img_format = ['jpg', 'png', 'jpeg']
    h = 0
    hist_ = np.zeros((3,256,1))
    for frmt in img_format:
        for img_address in glob.glob(directory + '*.' + frmt):      #using glob package to go to directory path
            img = cv.imread(img_address)
            img = cv.resize(img, (200,200))
            img_name = img_address.replace(directory, '')
            for x in range(3):
                hist_[x] = cv.calcHist([img], [x], None, [256], [0, 256])
            hist = cv.normalize(hist_, hist_)
            hist_b, hist_g, hist_r = np.zeros(256), np.zeros(256), np.zeros(256)
            for i in range(256):
                hist_b[i], hist_g[i], hist_r[i] = hist[0][i][0], hist[1][i][0], hist[2][i][0]
            df = pd.DataFrame({'name': img_name, 'Blue': hist_b, 'Green': hist_g, 'Red': hist_r})
            if h == 0:
                df.to_csv(directory + name + '.csv', index=False)
                h = 1
            else:
                df.to_csv(directory + name + '.csv', mode='a', header=False, index=False)
'''
directory_ = 'C:\\Users\\hanee\\Desktop\\Master\\Python\\Project\\Dataset\\'
name_ = 'dataset1'
histograms(directory_, name_)
'''

dataset_hist = pd.read_csv('C:\\Users\\hanee\\Desktop\\Master\\Python\\Project\\Dataset\\dataset.csv')

data_num = int(len(dataset_hist)/256)
#print(data_num)
#img = cv.imread('C:\\Users\\hanee\\Desktop\\Master\\Python\\Project\\Dataset\\2173.jpg')
img = cv.imread('C:\\Users\\hanee\\Desktop\\Master\\Python\\Project\\johann_lurf_starfilm08.jpg')
img = cv.resize(img, (200,200))
hist_n = np.zeros((3,256,1))
for c in range(3):
    hist_n[c] = cv.calcHist([img], [c], None, [256], [0, 256])
hist = cv.normalize(hist_n, hist_n)
j = 0
s = 0
dist = np.zeros((data_num,3,256))
for i in range(len(dataset_hist)+1):
    if i > 0 and i % 256 == 0:
        if s > (data_num): break
        for k in range(1,4):
            num = dataset_hist.iloc[j:i, k]
            for l in range(j,i):
                m = l-(256*(s))
                dist[s][k-1][m] = abs(num[l] - hist[k-1][m][0])
        j = i
        s += 1

distance = np.zeros(data_num)

for x in range(data_num):
    distance[x] = sum(dist[x][0]) + sum(dist[x][1]) + sum(dist[x][2])

indices = np.argsort(distance)

images_10 = indices[:10]
image_names = []
names = dataset_hist.iloc[0:(len(dataset_hist)+1):256, 0]
for z in range(data_num):
    image_names.append(names[z*256])

img1_ = []
img1 = []
for im in images_10:
    img1_.append('C:\\Users\\hanee\\Desktop\\Master\\Python\\Project\\Dataset\\' + image_names[im])

for h in range(10):
    img1.append(cv.resize(cv.imread(img1_[h]),(200,200)))

img_h1 = np.hstack((img1[0], img1[1], img1[2],img1[3], img1[4]))
img_h2 = np.hstack((img1[5], img1[6], img1[7],img1[8], img1[9]))
img_vh = np.vstack((img_h1,img_h2))

cv.imshow('image1', img_vh)
cv.waitKey(0)
