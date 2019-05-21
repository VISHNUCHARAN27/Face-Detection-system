# We first create the dataset
# Then we train the dataset
#Then we detect the data in the current frame
import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml") # A trained model for face detection
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml") # A trained model for eye detection

cap=cv2.VideoCapture(0)
id=input('Enter user-id:')
sampleNum=0


while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #We convert it to grayscale image because classifier works only with gray scale image
    faces=face_cascade.detectMultiScale(gray)  # This will detect all the faces in the current frame and will return the co-ordinates of the frame.
    for x,y,w,h in faces:
        sampleNum=sampleNum+1
        cv2.imwrite('\\'+'User.'+ str(id) +'.'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for ex,ey,ew,eh in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.waitKey(100)
    cv2.imshow('Image',frame)
    cv2.waitKey(1)
    if(sampleNum>20):
        break
cv2.destroyAllWindows()
cap.release()

