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

BH = math.floor(H / 60)
BW = math.floor(W / 60)
LH = 0
RH = BH 
print(BH,BW)

img  = np.zeros((H,W),dtype = np.uint8)


def get_block(Lx,Rx,Ly,Ry):
    img0 = imageA[Lx:Rx,Ly:Ry]
    img1 = imageB[Lx:Rx,Ly:Ry]
    img0_h_B = cv2.calcHist([img0], [0], None, [256], [0, 255])
    img1_h_B = cv2.calcHist([img1], [0], None, [256], [0, 255])
    img0_h_G = cv2.calcHist([img0], [1], None, [256], [0, 255])
    img1_h_G = cv2.calcHist([img1], [1], None, [256], [0, 255])
    img0_h_R = cv2.calcHist([img0], [2], None, [256], [0, 255])
    img1_h_R = cv2.calcHist([img1], [2], None, [256], [0, 255])
    similarity_b = cv2.compareHist(img0_h_B, img1_h_B, cv2.HISTCMP_CORREL)
    similarity_g = cv2.compareHist(img0_h_G, img1_h_G, cv2.HISTCMP_CORREL)
    similarity_r = cv2.compareHist(img0_h_R, img1_h_R, cv2.HISTCMP_CORREL)
    # print(similarity_b,similarity_g,similarity_r)
    if(min(similarity_b,similarity_g,similarity_r) < 0.08):
        img[Lx:Rx,Ly:Ry] = 255 * np.ones((Rx - Lx,Ry - Ly),dtype = np.uint8)
    # cv2.imshow('A ',img0)
    # cv2.imshow('B ',img1)    
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