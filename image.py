import numpy as np
import cv2

img = cv2.imread("japan.jpg")

cv2.imshow('japan',img)

img2 = img[50:50,50:50]

lower_red = np.array([45,350,230])
upper_red = np.array([255,255,255])

hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, lower_red, upper_red)

res = cv2.bitwise_and(img,img, mask= mask)

print(img.shape)
print(img[113,120])

cv2.imshow("no-mask", img)
cv2.imshow("mask", res)
cv2.imshow("img2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()