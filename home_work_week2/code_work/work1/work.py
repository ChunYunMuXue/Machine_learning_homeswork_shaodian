from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2

imageA = cv2.imread("D:\\article and  study\study\SELF_STUDYING\machine_learning_Dian\home_work_week2\code_work\pic\\1.webp")
imageB = cv2.imread("D:\\article and  study\study\SELF_STUDYING\machine_learning_Dian\home_work_week2\code_work\pic\\2.webp") # 读取图片

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

(score, diff) = compare_ssim(grayA, grayB, full=True)

diff = (diff * 255).astype("uint8")

thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # 二值化
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cnts = imutils.grab_contours(cnts)# 找轮廓

for c in cnts:              
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)# 画图
 
# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("thresh",thresh)
cv2.waitKey(0)
