import sys
import os
import cv2
from PIL import Image

directory ='namanww'
imagecount = 2000
faceCascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
os.makedirs(directory, exist_ok=True)

video = cv2.VideoCapture(0)

filename = len(os.listdir(directory))
count = 0

while True and count < imagecount:
    filename += 1
    count += 1
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5 ,30)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255, 255), 2)
        frame1=frame[y:y+h,x:x+h]
        im=Image.fromarray(frame1,'RGB')
        im = im.resize((32,32))
        im.save(os.path.join(directory, str(filename)+".jpg"), "JPEG")
        #cv2.imwrite(os.path.join(directory, str(filename)+".jpg"), im)
    cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
            break
video.release()
cv2.destroyAllWindows()