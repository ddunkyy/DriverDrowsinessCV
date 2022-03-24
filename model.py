import os
from tensorflow import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

print(os.curdir)

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
#print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(4, activation='softmax')
])
'''
for i in range(1):
    print("Loop #: " + str(i))
    model = load_model('models/cnncat2.h5')
    opt = keras.optimizers.Adam(learning_rate=0.01)

    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
    history1 = model.fit(train_batch, validation_data=valid_batch,epochs=100,steps_per_epoch=SPE ,validation_steps=VS)

    model.save('models/cnnCat2.h5', overwrite=True)
'''

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
history1 = model.fit(train_batch, validation_data=valid_batch,epochs=20,steps_per_epoch=SPE ,validation_steps=VS)

#print(history1.history.keys())

loss_train = history1.history['loss']
loss_val = history1.history['accuracy']
epochs = range(1,21)
plt.plot(epochs, loss_train, 'g', label='Loss')
plt.plot(epochs, loss_val, 'b', label='Accuracy')
plt.title('Model''s Accuracy and Loss')
plt.xlabel('Epochs')
plt.ylabel('Percentage')
plt.legend()
plt.show()