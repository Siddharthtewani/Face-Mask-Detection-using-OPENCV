#%%
import cv2 , datetime , os , warnings ,glob , random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import cvlib as cv
from tensorflow.keras.preprocessing.image import img_to_array
print("Done")
#%%
model=load_model(r'C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\notebook\model_new.model')

print("Done")
# %%

webcam=cv2.VideoCapture(0)
classes=["Mask","No Mask"]

while webcam.isOpened():
    status,frame=webcam.read()
    face,confidence = cv.detect_face(frame)
    

    for idx, f in enumerate(face):
        start_x,start_y=f[0],f[1]
        end_x,end_y=f[2],f[3]
        cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),(0,0,255),2)
        print(f)
        
        
        new_frame=cv2.resize(frame,(96,96))
        new_frame=new_frame.astype("float")/255.0
        pred_input=img_to_array(new_frame)
        pred_input=np.expand_dims(pred_input,axis=0)

        conf = model.predict(pred_input)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
        print(conf)
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        
        Y = start_y - 10 if start_y - 10 > 10 else start_y + 10

       
        
        cv2.putText(frame, label, (start_x+10,start_y-10),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0, 0), 2)

        cv2.imshow("Video",frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break    
cv2.waitKey()
cv2.destroyAllWindows()

#%%