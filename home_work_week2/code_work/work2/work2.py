from skimage import feature as ft
from skimage import data,filters
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2
import numpy as np
import math

imageA = cv2.imread("D:\\article and  study\study\SELF_STUDYING\machine_learning_Dian\home_work_week2\code_work\pic\\1.webp")
imageB = cv2.imread("D:\\article and  study\study\SELF_STUDYING\machine_learning_Dian\home_work_week2\code_work\pic\\2.webp")

H,W,R = imageA.shape
print(len(imageA),len(imageA[0]),len(imageA[0][0]))

BH = math.floor(H / 20)
BW = math.floor(W / 20)
LH = 0
RH = BH 
print(BH,BW)

img  = np.zeros((H,W),dtype = np.uint8)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)


def get_block(Lx,Rx,Ly,Ry):
    blockA = grayA[Lx:Rx,Ly:Ry]
    blockB = grayB[Lx:Rx,Ly:Ry]
    # grayA = filters.gaussian(grayA,sigma = 5)
    # grayB = filters.gaussian(grayB,sigma = 5)
    fA,featuresA = ft.hog(blockA,orientations=9,pixels_per_cell=[10,10],cells_per_block=[3,3],visualize=True,feature_vector = True)
    fB,featuresB = ft.hog(blockB,orientations=9,pixels_per_cell=[10,10],cells_per_block=[3,3],visualize=True,feature_vector = True)
    # cv2.imshow('A ',blockA)
    # cv2.imshow('B ',blockB)
    # cv2.imshow('A image',featuresA)
    # cv2.imshow('B image',featuresB)
    fc = (featuresA == featuresB)
    cnt = 0
    for x in fc:
        for y in x: 
            if y == (bool)(False):
                cnt += 1
    # print(cnt)
    # print(cnt / len(fc))
    # print(featuresA)
    if cnt  > 300:
        # print(Lx,Rx,Ly,Ry)
        (score, diff) = compare_ssim(blockA, blockB, full=True)
        print(score)
        if(score < 0.9):
            img[Lx:Rx,Ly:Ry] = 255 * np.ones((Rx - Lx,Ry - Ly),dtype = np.uint8)
    # cv2.waitKey(0) 

while(1):
    LW = 0
    RW = BW
    while(1):
        if(LW >= RW):break
        get_block(LH,RH,LW,RW)
        LW = RW
        RW += BW
        RW = min(RW,W - 1)
    LH = RH
    RH += BH
    RH = min(RH,H - 1)
    if(LH >= RH):break

cv2.imshow('ans image',img)
cv2.waitKey(0) # 创建答案记录背景版

cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts)# 找轮廓

for c in cnts:              
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)# 画图

cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.waitKey(0)