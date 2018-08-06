import numpy as np
import cv2



def morse_panel(data):
    size = np.shape(data[2])
    lebar_frame = 30
    
    
    frame = np.zeros((size[0]*lebar_frame,640,3),np.uint8)
    
    for index in range(size[0]):
        n = 0
        
        data_sementara = data[2][index]
        
        size_dta = len(data_sementara)
        
        if size_dta>630:
            data_sementara=data_sementara[size_dta-630:size_dta]
        
        for i in data_sementara:
            frame[(index+1)*lebar_frame-(i*20)-5][n]=[255,255,255]
            n=n+1
        frame[(index)*lebar_frame][:]=[15*index,45*index,65*index]
        
    return frame
            
        


cap = cv2.VideoCapture(1)

thr_value = 230

kernel =  np.zeros((5,5),np.uint8)

x,y=np.shape(kernel)


circ_err = 10
max_area = 20

kernel[:,x//2]=1
kernel[x//2,:]=1


data = [[],[],[]]


font = cv2.FONT_HERSHEY_SIMPLEX


while(True):
    # Capture frame-by-frame
    
    
    ret, frame = cap.read()
    if ret == True:
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        ret,thresh1 = cv2.threshold(gray,thr_value,255,cv2.THRESH_BINARY)
        
        
#        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#        
#        lower_blue = np.array([110,50,50])
#        upper_blue = np.array([130,200,200])
#        
#        
#        
#        thresh1 = cv2.inRange(hsv, lower_blue, upper_blue)
#        

           
    
        thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    
    
    
        image, contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        
        for c in contours:
    
            x1,y1,w,h = cv2.boundingRect(c)
            x2=x1+w
            y2=y1+h
            
            
                
            
            if np.abs(np.subtract(w,h))<5:
                
                if w >max_area:
                    
#                    frame = cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                    
                    size = np.shape(data[0])
                    
                    if size==0:
                        data[0].append([x1,y1,x2,y2])
                        data[1].append(1)
                        data[2].append([])
                        
                    else :
                        
                        sinyal = False
                        for index in range(size[0]):

                            if ((data[0][index][0]<x1+(w//2)<data[0][index][2]  and (data[0][index][1]<y1+(h//2)<data[0][index][3]))):
                                sinyal = True
                                
                                
                                
                                data[0][index]=[x1,y1,x2,y2]
                                data[1][index]=data[1][index]+1




#                            if ((data[0][index][0]<x1<data[0][index][2] or data[0][index][0]<x2<data[0][index][2]) and (data[0][index][1]<y1<data[0][index][3] or data[0][index][1]<y2<data[0][index][3])):
#                                sinyal = True
#                                
#                                
#                                
#                                data[0][index]=[x1,y1,x2,y2]
#                                data[1][index]=data[1][index]+1
                        if sinyal== False:
                            data[0].append([x1,y1,x2,y2])
                            data[1].append(1)
                            data[2].append([])
                        
                        
#                    size = np.shape(data[0])
                        
#        print (sinyal)            
        size = np.shape(data[0])                
        
        for index in range(size[0]):
            
            poss = data[0][index]
            
            frame = cv2.rectangle(frame,(poss[0],poss[1]),(poss[2],poss[3]),(0,255,0),3)
            
            
            if np.sum(thresh1[poss[0]:poss[2]][poss[1]:poss[3]])>0:
                data[2][index].append(1)
            else:
                data[2][index].append(0)
                
            
            
            
                        
                        
            cv2.putText(frame,str(data[1][index]//30),(poss[0],poss[3]), font, 2,(0,0,255),2,cv2.LINE_AA)



                    
                            
        cv2.putText(frame,"number of object = "+str(size[0]),(10,200), font, 1,(255,0,0),2,cv2.LINE_AA)  
                        
                            
    
    
        morse = morse_panel(data)
    
    
        cv2.imshow('frame',frame)
        cv2.imshow('mask',thresh1)
        
        
        if size[0]>0:
            
            cv2.imshow('mask',morse)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
