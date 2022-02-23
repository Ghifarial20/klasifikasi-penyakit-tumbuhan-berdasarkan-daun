# -*- coding: utf-8 -*-
"""
Created on Sun May 31 13:28:26 2020

@author: fachr
"""
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import cv2 as cv

radius = 2
n_points = 8 * radius
METHOD = 'uniform'

x=0
training = []
responses = []
eps=1e-7

# Buka File dataset
for filename in glob.glob("Datasets/Bacteria/*jpg"):
    im = cv.imread(filename)
    hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    
    #Segmentasi warna pada dataset
    lower_blue = np.array([25, 20, 72])
    upper_blue = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10),np.uint8)
    se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
            
    dst_dilate = cv.dilate(mask, se_3, iterations = 1)
    dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
    dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
    dst_erosi2 = cv.erode(dst_dilate2, se_5,     iterations = 3)
    dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)
            
    res = cv.bitwise_and(im,im, mask= dst_dilate3)
            
    
    b,g,r = cv.split(res)
    rgb = cv.merge([r,g,b])
    for i in range(rgb.shape[0]):
       for j in range (rgb.shape[1]):
             if(rgb[i][j][0] == 0):
                 rgb[i][j][0] = 255
             if(rgb[i][j][1] == 0):
                 rgb[i][j][1] = 255
             if(rgb[i][j][2] == 0):
                 rgb[i][j][2] = 255
    gmbr = rgb.flatten()
    training.append(gmbr)
    responses.append(1)
    
for filename in glob.glob("Datasets/Fungi/*jpg"):
    im = cv.imread(filename)
    hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    #Range warna biru segmentasi/klasifikasi
    lower_blue = np.array([25, 20, 72])
    upper_blue = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10),np.uint8)
    se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
            
    dst_dilate = cv.dilate(mask, se_3, iterations = 1)
    dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
    dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
    dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
    dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)
            
    res = cv.bitwise_and(im,im, mask= dst_dilate3)
            
    #image = kontras_red(im)
    b,g,r = cv.split(res)
    rgb = cv.merge([r,g,b])
    for i in range(rgb.shape[0]):
       for j in range (rgb.shape[1]):
             if(rgb[i][j][0] == 0):
                 rgb[i][j][0] = 255
             if(rgb[i][j][1] == 0):
                 rgb[i][j][1] = 255
             if(rgb[i][j][2] == 0):
                 rgb[i][j][2] = 255
    gmbr = rgb.flatten()
    training.append(gmbr)
    responses.append(2)
    
for filename in glob.glob("Datasets/Nematodes/*jpg"):
    im = cv.imread(filename)
    hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    #Range warna biru segmentasi/klasifikasi
    lower_blue = np.array([25, 20, 72])
    upper_blue = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10),np.uint8)
    se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
            
    dst_dilate = cv.dilate(mask, se_3, iterations = 1)
    dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
    dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
    dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
    dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)
            
    res = cv.bitwise_and(im,im, mask= dst_dilate3)
            
    #image = kontras_red(im)
    b,g,r = cv.split(res)
    rgb = cv.merge([r,g,b])
    for i in range(rgb.shape[0]):
       for j in range (rgb.shape[1]):
             if(rgb[i][j][0] == 0):
                 rgb[i][j][0] = 255
             if(rgb[i][j][1] == 0):
                 rgb[i][j][1] = 255
             if(rgb[i][j][2] == 0):
                 rgb[i][j][2] = 255
    gmbr = rgb.flatten()
    training.append(gmbr)
    responses.append(3)

for filename in glob.glob("Datasets/Normal/*jpg"):
    im = cv.imread(filename)
    hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    #Range warna biru segmentasi/klasifikasi
    lower_blue = np.array([25, 20, 72])
    upper_blue = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10),np.uint8)
    se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
            
    dst_dilate = cv.dilate(mask, se_3, iterations = 1)
    dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
    dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
    dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
    dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)
            
    res = cv.bitwise_and(im,im, mask= dst_dilate3)
            
    #image = kontras_red(im)
    b,g,r = cv.split(res)
    rgb = cv.merge([r,g,b])
    for i in range(rgb.shape[0]):
       for j in range (rgb.shape[1]):
             if(rgb[i][j][0] == 0):
                 rgb[i][j][0] = 255
             if(rgb[i][j][1] == 0):
                 rgb[i][j][1] = 255
             if(rgb[i][j][2] == 0):
                 rgb[i][j][2] = 255
    gmbr = rgb.flatten()
    training.append(gmbr)
    responses.append(4)
    
for filename in glob.glob("Datasets/Virus/*jpg"):
    im = cv.imread(filename)
    hsv = cv.cvtColor(im,cv.COLOR_BGR2HSV)
    #Range warna biru segmentasi/klasifikasi
    lower_blue = np.array([25, 20, 72])
    upper_blue = np.array([102, 255, 255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    kernel = np.ones((10,10),np.uint8)
    se_3 = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    se_5 = cv.getStructuringElement(cv.MORPH_RECT,(5,5))
            
    dst_dilate = cv.dilate(mask, se_3, iterations = 1)
    dst_erosi = cv.erode(dst_dilate, se_3, iterations = 2)
    dst_dilate2 = cv.dilate(dst_erosi, se_5, iterations = 2)
    dst_erosi2 = cv.erode(dst_dilate2, se_5, iterations = 3)
    dst_dilate3 = cv.dilate(dst_erosi2, se_3, iterations = 1)
            
    res = cv.bitwise_and(im,im, mask= dst_dilate3)
            
    #image = kontras_red(im)
    b,g,r = cv.split(res)
    rgb = cv.merge([r,g,b])
    
    for i in range(rgb.shape[0]):
       for j in range (rgb.shape[1]):
             if(rgb[i][j][0] == 0):
                 rgb[i][j][0] = 255
             if(rgb[i][j][1] == 0):
                 rgb[i][j][1] = 255
             if(rgb[i][j][2] == 0):
                 rgb[i][j][2] = 255

    gmbr = rgb.flatten()
    training.append(gmbr)
    responses.append(5)

'''
0.5 200 = 76.67%
0.5 250 = 70.83
0.5 150 = 78.33
0.5 100 = 67.50
0.5 125 = 75.00
0.5 175 = 71.67
0.5 165 = 73.33
0.5 160 = 70.00

0.6 150 = 70.83
0.4 150 = 75.00
0.55 150 = 73.48
0.8 150 = 66.15

0.5 500 = 69.17
0.5 800 = 67.50
0.5 222 = 70.83
0.5 555 = 71.67
0.5 666 = 66.67
0.5 999 = 71.67
'''

#membagi data -> training dan test
x_train, x_test, y_train, y_test = train_test_split(training, responses, test_size = 0.5, random_state = 150)

#NAIVE BAYES
from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
gnb.fit(x_train, y_train)
 
#mendapatkan hasil prediksi
y_pred = gnb.predict(x_test)
print('')
print('======================================================================')
print('')
print('Naive BAYES :')
#menampilkan presentasi akurasi
error = ((y_test != y_pred).sum()/len(y_pred))*100
print('error prediksi = %.2f' %error, '%')
akurasi = 100-error
print('akurasi = %.2f' %akurasi, '%')