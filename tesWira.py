# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 14:52:36 2018

@author: sangs
"""

import numpy as np
import cv2

kernel = np.zeros((5,5),np.uint8)

print(kernel)

x,y = np.shape(kernel)
kernel[:,x//2]=1
kernel[x//2,:]=1


cap = cv2.VideoCapture(1)

while(1):
    ret,frame = cap.read()
    
    ret,thresh1 = cv2.threshold(frame,230,255,cv2.THRESH_BINARY)
        
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    cv2.imshow("thres",thresh1)
    cv2.imshow("Frame",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()