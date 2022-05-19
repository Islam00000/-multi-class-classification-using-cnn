#!/usr/bin/env python
# coding: utf-8
# dataset link : https://www.kaggle.com/datasets/ayushv322/animal-classification
# In[1]:


import numpy as np
import pandas as pd 
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn.utils import shuffle           
import matplotlib.pyplot as plt             
import cv2                                 
import tensorflow as tf                
from tqdm import tqdm
import pathlib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from PIL import Image 
from glob import glob


# In[2]:


G1_path = "H:\\Data sets\\Data\\G1"
G2_path = "H:\\Data sets\\Data\\G2"
G3_path = "H:\\Data sets\\Data\\G3"
test_path = "H:\\Data sets\\Data\\test"
G1_G2_path = "H:\\Data sets\\Data\\G1_G2"
G1_G3_path = "H:\\Data sets\\Data\\G1_G3"
G2_G3_path = "H:\\Data sets\\Data\\G2_G3"


# In[3]:


def read_image(folder: pathlib.PosixPath):
    name = []
    clas = []
    processed =[]
    processed = pd.DataFrame(processed)
    for img in folder.iterdir():
        if img.suffix == ".jpg":
            name.append(str(img.parts[-1]).split(".")[0])
            clas.append(folder.parts[-1])
            
    processed["class"] = clas 
    processed ["name"] = name
   
   
    
    return processed
    


# In[4]:


Buffalo = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G1\\Buffalo"))
Elephant = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G1\\Elephant"))                        
Rhino = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G1\\Rhino")) 
Zebra = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G1\\Zebra"))

G1 = pd.concat([Buffalo,Elephant,Rhino,Zebra],axis=0)
G1.to_csv('H:\\Data sets\\Data\\G1.csv',index=False)


# In[5]:


Buffalo = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G2\\Buffalo"))
Elephant = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G2\\Elephant"))                        
Rhino = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G2\\Rhino")) 
Zebra = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G2\\Zebra"))

G2 = pd.concat([Buffalo,Elephant,Rhino,Zebra],axis=0)
G2.to_csv('H:\\Data sets\\Data\\G2.csv',index=False)


# In[6]:


Buffalo = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G3\\Buffalo"))
Elephant = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G3\\Elephant"))                        
Rhino = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G3\\Rhino")) 
Zebra = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\G3\\Zebra"))

G3 = pd.concat([Buffalo,Elephant,Rhino,Zebra],axis=0)
G3.to_csv('H:\\Data sets\\Data\\G3.csv',index=False)


# In[7]:


Buffalo = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\test\\Buffalo"))
Elephant = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\test\\Elephant"))                        
Rhino = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\test\\Rhino")) 
Zebra = read_image(folder = pathlib.Path.cwd().joinpath("H:\\Data sets\\Data\\test\\Zebra"))

test = pd.concat([Buffalo,Elephant,Rhino,Zebra],axis=0)
test.to_csv('H:\\Data sets\\Data\\test.csv',index=False)


# In[8]:


file = pd.concat([G1,G2,G3,test],axis=0)
file.to_csv('H:\\Data sets\\Data\\Data.csv',index=False)


# In[9]:


class_names = ['Buffalo', 'Elephant', 'Rhino', 'Zebra']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

nb_classes = len(class_names)

IMAGE_SIZE = (100, 100)


# In[4]:


class_names_label


# In[10]:


def load_data(path):
    
    images = []
    labels = []
    for folder in os.listdir(path):
        label = class_names_label[folder]
        for file in tqdm(os.listdir(os.path.join(path, folder))):
           
                img_path = os.path.join(os.path.join(path, folder), file)
                
               
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, IMAGE_SIZE) 
                
               
                images.append(image)
                labels.append(label)
            
                       
    images = np.array(images, dtype = 'float32')
    labels = np.array(labels, dtype = 'int32')   
        
    

    return images,labels


# In[11]:


train_images,train_labels = load_data(G1_G2_path)


# In[12]:


val_images,val_labels = load_data(G3_path)


# In[13]:


test_images,test_labels = load_data(test_path)


# In[14]:


train = pd.concat([G1,G2],axis=0)
train["type"] = "train"
G3["type"] = "val"
train = pd.concat([train,G3],axis=0)
G3 = G3.drop(["type"],axis="columns")
sn.countplot(x="class", hue="type", data=train)


# In[15]:


sn.countplot(test["class"])
plt.title("Images of test")


# In[16]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
val_images, val_labels = shuffle(val_images, val_labels, random_state=25)


# In[17]:


train_images = train_images / 255.0 
val_images = val_images / 255.0
test_images = test_images / 255.0


# In[18]:


model = Sequential()
model.add(Conv2D(32, (3,3), input_shape= (100,100,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3),))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Conv2D(64, (3,3),))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(4))#output
model.add(Activation("softmax"))


# In[46]:


model.summary()


# In[19]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])


# In[20]:


history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=40,batch_size=50)


# In[21]:


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# In[22]:


test_loss = model.evaluate(test_images, test_labels)


# In[29]:


predictions = model.predict(test_images)    
pred_labels = np.argmax(predictions, axis = 1)

CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.xlabel("predict")
plt.ylabel("truth")
plt.show()


# In[28]:


cycle_one_report = classification_report(test_labels,pred_labels, target_names=class_names)
print(cycle_one_report)   


# In[30]:


train = pd.concat([G1,G3],axis=0)
train["type"] = "train"
G2["type"] = "val"
train = pd.concat([train,G2],axis=0)
G2 = G2.drop(["type"],axis="columns")
sn.countplot(x="class", hue="type", data=train)


# In[31]:


train_images,train_labels = load_data(G1_G3_path)


# In[32]:


val_images,val_labels = load_data(G2_path)


# In[33]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
val_images, val_labels = shuffle(val_images, val_labels, random_state=25)
train_images = train_images / 255.0 
val_images = val_images / 255.0


# In[34]:


history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15,batch_size=133)


# In[35]:


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# In[36]:


test_loss = model.evaluate(test_images, test_labels)


# In[38]:


predictions = model.predict(test_images)    
pred_labels = np.argmax(predictions, axis = 1)
CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.xlabel("predict")
plt.ylabel("truth")
plt.show()


# In[39]:


cycle_two_report = classification_report(test_labels,pred_labels, target_names=class_names)
print(cycle_two_report)    


# In[42]:


train = pd.concat([G2,G3],axis=0)
train["type"] = "train"
G1["type"] = "val"
train = pd.concat([train,G1],axis=0)
G1 = G1.drop(["type"],axis="columns")
sn.countplot(x="class", hue="type", data=train)


# In[43]:


train_images,train_labels = load_data(G2_G3_path)


# In[44]:


val_images,val_labels = load_data(G1_path)


# In[45]:


train_images, train_labels = shuffle(train_images, train_labels, random_state=25)
val_images, val_labels = shuffle(val_images, val_labels, random_state=25)
train_images = train_images / 255.0 
val_images = val_images / 255.0


# In[46]:


history = model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=15,batch_size=133)


# In[47]:


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()


# In[48]:


test_loss = model.evaluate(test_images, test_labels)


# In[40]:


predictions = model.predict(test_images)    
pred_labels = np.argmax(predictions, axis = 1)
CM = confusion_matrix(test_labels, pred_labels)
ax = plt.axes()
sn.heatmap(CM, annot=True, 
           annot_kws={"size": 10}, 
           xticklabels=class_names, 
           yticklabels=class_names, ax = ax)
ax.set_title('Confusion matrix')
plt.xlabel("predict")
plt.ylabel("truth")
plt.show()


# In[50]:


cycle_three_report = classification_report(test_labels,pred_labels, target_names=class_names)
print(cycle_three_report)    


# In[42]:


precision = {'subset1':[0.79, 0.65, 0.76, 0.98],
        'subset2':[0.74, 0.80, 0.78, 0.96],
            "subset3" : [0.77,0.80,0.76,0.95]}
precision = pd.DataFrame(precision, index =['Buffalo','Elephant','Rhino','Zebra'])
precision["average"] = (precision["subset1"] + precision["subset2"] + precision["subset3"]) / 3
precision


# In[43]:


recall = {'subset1':[0.65, 0.82, 0.74, 0.92],
        'subset2':[0.80, 0.72, 0.81, 0.93],
            "subset3" : [0.77,0.75,0.82,0.94]}
recall  = pd.DataFrame(recall , index =['Buffalo','Elephant','Rhino','Zebra'])
recall["average"] = (recall ["subset1"] + recall["subset2"] + recall["subset3"]) / 3
recall 


# In[45]:


accuracy = {'subset1':[0.71, 0.73, 0.75, 0.95],
        'subset2':[0.77, 0.76, 0.79, 0.95],
            "subset3" : [0.77,0.77,0.79,0.94]}
accuracy = pd.DataFrame(accuracy , index =['Buffalo','Elephant','Rhino','Zebra'])
accuracy["average"] = (accuracy["subset1"] + accuracy["subset2"] + accuracy["subset3"]) / 3
accuracy


# In[ ]:




