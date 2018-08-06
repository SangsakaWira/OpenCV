import numpy as np
import cv2

cap = cv2.VideoCapture(1)

thr_value = 100

kernel =  np.zeros((5,5))

x,y=np.shape(kernel)

kernel[:,x//2]=1
kernel[x//2,:]=1


while(True):
    # Capture frame-by-frame
    
    
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret,thresh1 = cv2.threshold(gray,thr_value,255,cv2.THRESH_BINARY_INV)


    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)



    image, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)











    cv2.imshow('frame',frame)
    cv2.imshow('mask',thresh1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
