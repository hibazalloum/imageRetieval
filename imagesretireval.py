import cv2 as cv
import numpy as np
import glob
import pandas as pd


def dataSetHistograms(path, name):
    img_format = ['jpg', 'png', 'jpeg']
    # H : variable to save histogram for labels on cvs
    h = 0
    hist = np.zeros((3, 256, 1))
    for frmt in img_format:
        for img_address in glob.glob(path + '*.' + frmt):  # using glob package to go to directory path
            img = cv.imread(img_address)
            img = cv.resize(img, (200, 200))

            # to take image name without path
            img_name = img_address.replace(path, '')

            # calculate Histogram for BGR Color
            for color in range(3):
                hist[color] = cv.calcHist([img], [color], None, [256], [0, 256])
            hist = cv.normalize(hist, hist)
            hist_b, hist_g, hist_r = np.zeros(256), np.zeros(256), np.zeros(256)

            # To save data as a number not strings
            for i in range(256):
                hist_b[i], hist_g[i], hist_r[i] = hist[0][i][0], hist[1][i][0], hist[2][i][0]

            # save data as csv
            df = pd.DataFrame({'name': img_name, 'Blue': hist_b, 'Green': hist_g, 'Red': hist_r})
            if h == 0:
                df.to_csv(path + name + '.csv', index=False)
                h = 1
            else:
                df.to_csv(path + name + '.csv', mode='a', header=False, index=False)


def histQueryImage(path):
    query_image = cv.imread(path)
    query_image = cv.resize(query_image, (200, 200))
    hist_n = np.zeros((3, 256, 1))
    for c in range(3):
        hist_n[c] = cv.calcHist([query_image], [c], None, [256], [0, 256])
    hist = cv.normalize(hist_n, hist_n)
    return hist


def distanceHistogram(dataNum, dataSetHist):
    j = 0
    pointer = 0
    # create Array for Distances
    dist = np.zeros((dataNum, 3, 256))  # number of image on the dataset
    for i in range(len(dataSetHist) + 1):
        if i > 0 and i % 256 == 0:  # To prevent redundancy 256 for one image ONLY ONE PERIOD
            if pointer > (dataNum):
                break

            for colors in range(1, 4):
                num = dataSetHist.iloc[j:i, colors]  # to take colors of image columns
                for value in range(j, i):  # to take value of colors columns
                    m = value - (256 * (pointer))

                    # save distance and differance between degree of colors and histogram image query
                    dist[pointer][colors - 1][m] = abs(num[value] - histQuery[colors - 1][m][0])
            j = i
            pointer += 1

    distance = np.zeros(dataNum)
    # sum distance
    for x in range(dataNum):
        distance[x] = sum(dist[x][0]) + sum(dist[x][1]) + sum(dist[x][2])
    indices = np.argsort(distance)  # ascending order
    return indices


def topTenImages(path):
    #  takes name of the image once
    image_names = []
    names = dataSetHist.iloc[0:(len(dataSetHist) + 1):256, 0]

    # create new array Takes names of images with index = zero for number of image
    for z in range(dataNum):
        image_names.append(names[z * 256])

    # save the path of similar images to read and present them
    img1_ = []
    img1 = []
    for im in top_ten:
        img1_.append(path + image_names[im])
    for h in range(10):
        img1.append(cv.resize(cv.imread(img1_[h]), (200, 200)))

    # to view result in one window!
    img_h1 = np.hstack((img1[0], img1[1], img1[2], img1[3], img1[4]))
    img_h2 = np.hstack((img1[5], img1[6], img1[7], img1[8], img1[9]))
    img_vh = np.vstack((img_h1, img_h2))

    cv.imshow('image1', img_vh)
    cv.waitKey(0)


if __name__ == '__main__':
    data_set_path = "Dataset\\"
    name_path ="dataset"
    dataSetHistograms(data_set_path, name_path)
    dataSetHist = pd.read_csv("Dataset\\dataset.csv")
    dataNum = int(len(dataSetHist) / 256)
    histQuery = histQueryImage("Dataset\\3551.jpg")
    distance = distanceHistogram(dataNum, dataSetHist)
    top_ten = distance[:10]
    view_result = topTenImages("Dataset/")
