#%%
import cv2 , datetime , os , warnings ,glob , random ,re
import pandas as pd
import numpy as np 
from  numpy import expand_dims
import matplotlib.pyplot as plt
# import seaborn as sns 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout, Dense


print("Importing Done")
#%%
input_size=(96,96,3)
image_files=[]
labels=[]
data=[]
#%%
folders=glob.glob(r"C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\Datasets\New Masks Dataset\Train\Mask\*")
#%%

# for i in glob.glob(r"C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\Datasets\New Masks Dataset\Train\**\*"):
#     image_files.append(i)  
#     random.shuffle(image_files)


# #%%
# for img in image_files:
#     lab=img.split(os.path.sep)[-2]
#     if (lab=="Non Mask"):
#         labels.append(0)
    
#     elif(lab=="Mask"):
#         labels.append(1)


#     image=cv2.imread(img)
#     image_resized=cv2.resize(image,(96,96))
#     img_array=img_to_array(image_resized)/255.0
#     data.append(img_array)
# #%%
# data=np.array(data,dtype=np.float64)/255.0
# labels=np.array(labels,dtype=np.int64)
# #%%
# print(data.shape)
# print(labels.shape)  





# #%%
# # labels_final=to_categorical(labels,num_classes=2)
# final_labels=labels.reshape(-1,1)
# #%%
# print(final_labels.shape)



# # %%
# no_data=0
# label_no=0
# for img in image_files:
#     lab=img.split(os.path.sep)[-2]

#     if (lab=="Mask"):
#         labels.append("1")
#         label_no+=1
#     elif (lab=="Non Mask"):
        
#         labels.append("0")
#         label_no+=1

#     print(lab)
#     img=cv2.imread(img)
#     image=cv2.resize(img,(244,244))
#     img_array=img_to_array(image)/255.0
#     print(img_array.shape) 
#     no_data+=1
#     data.append(img_array)
# final_image = np.expand_dims(data, axis=0) 
# train_y=to_categorical(labels,num_classes=2)
# print(no_data)
# print(label_no)
# %%
# # %%
image_gene=ImageDataGenerator(featurewise_center=False,
                             samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            
                            rotation_range=45,
                            width_shift_range=0.3,
                            height_shift_range=0.3, 
                            brightness_range=None, 
                            
                            zoom_range=0.2, 
                            channel_shift_range=0.0, 
                            fill_mode='nearest', 
                            cval=0.0, 
                            horizontal_flip=True,
                            vertical_flip=True, 
                            rescale=None,
                            preprocessing_function=None, 
                            data_format=None, 
                            validation_split=0.0
                            , dtype=None  )
# %%
# for img in image_files:
#     image=load_img(img)
#     x=img_to_array(image)
#     x=x.reshape((1,)+x.shape)


#     lab=img.split(os.path.sep)[-2]
#     if (lab=="Non Mask"):
#         i=0
#         for batch in image_gene.flow(x,batch_size=1,save_to_dir=r"C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\Datasets\augumented\Non Mask"):
#             i+=1
#             if i>10: 
#                 break
    
#     elif(lab=="Mask"):
#         i=0
#         for batch in image_gene.flow(x,batch_size=1,save_to_dir=r"C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\Datasets\augumented\Mask"):
#             i+=1
#             if i>10: 
#                 break

#%%
input_size=(96,96,3)
image_files=[]
labels=[]
data=[]
#%%
for i in glob.glob(r"C:\Users\siddh\Desktop\Projects\Face_Mask_Detector\Datasets\augumented\**\*"):
    image_files.append(i)  
    random.shuffle(image_files)


#%%
for img in image_files:
    lab=img.split(os.path.sep)[-2]
    if (lab=="Non Mask"):
        labels.append(0)
    
    elif(lab=="Mask"):
        labels.append(1)


    image=cv2.imread(img)
    image_resized=cv2.resize(image,(96,96))
    img_array=img_to_array(image_resized)/255.0
    data.append(img_array)
#%%
data=np.array(data,dtype=np.float64)/255.0
labels=np.array(labels,dtype=np.int64)
#%%
print(data.shape)
print(labels.shape)  

final_labels=labels.reshape(-1,1)
#%%
x_train,y_test,y_train,y_test=train_test_split(data,labels,final_labels,test_split=0.1)




# %%

def bulid_model(height, width, depth):
    model=keras.Sequential()
    
    model.add(Conv2D(200,kernel_size=(3,3),input_shape=(height,width,depth)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))
    

    model.add(Conv2D(200,kernel_size=(3,3)))
    model.add(Activation("relu"))
    

    model.add(Conv2D(200,kernel_size=(3,3)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3,3)))



    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(BatchNormalization())

    model.add(Dense(200))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    

    model.add(Dense(1))
    model.add(Activation("softmax"))


    return model

#%%
model= bulid_model(width=input_size[0], height=input_size[1], depth=input_size[2])
model.summary()
# %%
opt=keras.optimizers.SGD(learning_rate=0.001)
loss=keras.losses.BinaryCrossentropy()
model.compile(optimizer=opt, loss=loss, metrics=["mae"])
# %%
# train_=np.array(train_y)

model.fit(data,final_labels,epochs=20,batch_size=10)
# %%
# model.save("model_new.model")
# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
