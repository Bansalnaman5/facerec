import os
from PIL import Image
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
from keras.models import Sequential
from keras import models
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import cv2
from keras import optimizers
import h5py
import time
import pickle
li=os.listdir("C:\\Users\\Naman\\Desktop\\python\\pics")
global cur
curr=max(li)
curr=int(float(curr))
cur=curr
print(cur)
curr=str(curr+1)
def ret_hot(i):
    print(cur+1)
    s=[0 for k in range(cur+1)]
    s[i]=1
    return s
print("already exist in db ??  y/n (case sensitive)")
n=input()
if(n=='n'):
    k=input('enter name :')
    d=pd.read_csv("C:\\Users\\Naman\\Desktop\\python\\facenames.csv")
    names=d['0']
    al=[]
    for i in names.values:
        al.append(i)
    al.append(k)
    a=pd.DataFrame(al)
    a.to_csv('facenames.csv')
    print('get ready for pictures')
    time.sleep(2)
    cur=cur+1

    directory =str("pics")+ str("\\")+curr
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

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (2,255, 2), 2)
            frame1=frame[y:y+h,x:x+h]
            im=Image.fromarray(frame1,'RGB')
            im = im.resize((32,32))
            im.save(os.path.join(directory, str(filename)+".jpg"), "JPEG")
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
    pic=os.listdir("C:\\Users\\Naman\\Desktop\\python\\pics")
    imag=[]
    res=[]
    for  i in pic:
        f=os.listdir("C:\\Users\\Naman\\Desktop\\python\\pics\\"+str(i))
        for j in f[300:1800]:
            image=Image.open("C:\\Users\\Naman\\Desktop\\python\\pics\\"+str(i)+'\\'+str(j))
            data=np.asarray(image)
            d=data.ravel()
            imag.append(d)
            res.append(ret_hot(int(float(i))))
    imag,res=shuffle(imag,res,random_state=0)
    imag=np.array(imag)
    res=np.array(res)
    imag=imag.reshape(-1,32,32,3)

    model=Sequential()
    model.add(Conv2D(135,(2,2),activation='sigmoid',input_shape=(32,32,3)))

    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(500,activation='relu'))#,input_shape=(32,32,3)))
    model.add(Dense(250,activation='softplus'))
    model.add(Dense(cur+1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['mae'])
    model.fit(imag,res,epochs=5,batch_size=200)
    model.save("face.h5")
    vid=cv2.VideoCapture(0)
    faceCascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        u,frame=vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5 ,30)
        
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 25), 2)
            frame1=frame[y:y+h,x:x+h]
            im=Image.fromarray(frame1,'RGB')
            im = im.resize((32,32))
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            #print(img_array)
            pre=model.predict(img_array)
            a=np.where(pre[0]==np.amax(pre[0]))
            #print(round(pre[0][1],2))
            p=a[0][0]
            if (np.amax(pre[0]))>=0.5:
                cv2.putText(frame, al[p],(x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 0))
            else:
                cv2.putText(frame,"Unidentified",(x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 0))

            
        cv2.imshow("optimizing",frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()    
else:
    vid=cv2.VideoCapture(0)
    faceCascade  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model=models.load_model('face.h5')
    c=0
    d=pd.read_csv("C:\\Users\\Naman\\Desktop\\python\\facenames.csv")
    names=d['0']
    while True:
        u,frame=vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.3, 5 ,30)
        
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 2)
            frame1=frame[y:y+h,x:x+h]
            im=Image.fromarray(frame1,'RGB')
            im = im.resize((32,32))
            img_array = np.array(im)
            img_array = np.expand_dims(img_array, axis=0)
            #print(img_array)
            pre=model.predict(img_array)
            a=np.where(pre[0]==np.amax(pre[0]))
            #print(round(np.amax(pre[0]),2))
            p=a[0][0]
            if (np.amax(pre[0]))>=0.5:
                cv2.putText(frame, names[p],(x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 0))
            else:
                cv2.putText(frame,"Unidentified",(x,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 0))

        cv2.imshow("optimizing",frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
