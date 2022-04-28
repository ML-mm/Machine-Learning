import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_datagen = ImageDataGenerator(rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')


#train_datagen.fit(x_train)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

train_datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.0001)))
model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)))
model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test),
        callbacks = [EarlyStopping(monitor= 'val_accuracy', patience = 2)])
test_loss, test_acc = model.evaluate(x_test, y_test)

image_no = 1
while os.path.isfile(f"numbers/{image_no}.png"):
    try:
        img = cv2.imread(f"numbers/{image_no}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is most likely {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error: check code again.")
    finally:
        image_no += 1


#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=20)
#model.save('handwriting3.model')

'''
model = tf.keras.models.load_model('handwriting3.model')

image_no = 1
while os.path.isfile(f"numbers/{image_no}.png"):
    try:
        img = cv2.imread(f"numbers/{image_no}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is most likely {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error: check code again.")
    finally:
        image_no += 1
'''

#loss, accuracy = model.evaluate(x_test, y_test)

#print(loss, accuracy)

#x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
#input_shape = (28, 28, 1)
