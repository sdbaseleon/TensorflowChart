import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import cv2

import pickle

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(16))

model.add(Dense(1))
model.add(Activation('sigmoid'))

from tensorflow.python.keras.optimizers import SGD
opt = SGD(lr=0.01)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(X, y, batch_size=4, epochs=15, validation_split=0.3, verbose=1)

model.save('cnn_futures_chart.h5')

CATEGORIES = ["Bull", "Bear"]

def prepare(filepath):
    IMG_SIZE = 70  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (599, 280))
    return new_array.reshape(-1, 599, 280, 1)

prediction = model.predict([prepare('_test_trades\\0.jpg')])

prediction1 = model.predict([prepare('_test_trades\\157.jpg')])
prediction2 = model.predict([prepare('_test_trades\\158.jpg')])
prediction3 = model.predict([prepare('_test_trades\\159.jpg')])
prediction4 = model.predict([prepare('_test_trades\\160.jpg')])
prediction5 = model.predict([prepare('_test_trades\\161.jpg')])
prediction6 = model.predict([prepare('_test_trades\\162.jpg')])
prediction7 = model.predict([prepare('_test_trades\\163.jpg')])
prediction8 = model.predict([prepare('_test_trades\\164.jpg')])
prediction9 = model.predict([prepare('_test_trades\\213.jpg')])
prediction10 = model.predict([prepare('_test_trades\\214.jpg')])
prediction11 = model.predict([prepare('_test_trades\\215.jpg')])
prediction12 = model.predict([prepare('_test_trades\\216.jpg')])
prediction13 = model.predict([prepare('_test_trades\\217.jpg')])
prediction14 = model.predict([prepare('_test_trades\\218.jpg')])
prediction15 = model.predict([prepare('_test_trades\\219.jpg')])
prediction16 = model.predict([prepare('_test_trades\\220.jpg')])
prediction17 = model.predict([prepare('_test_trades\\221.jpg')])
prediction18 = model.predict([prepare('_test_trades\\222.jpg')])
prediction19 = model.predict([prepare('_test_trades\\223.jpg')])
prediction20 = model.predict([prepare('_test_trades\\165.jpg')])
prediction21 = model.predict([prepare('_test_trades\\166.jpg')])
prediction22 = model.predict([prepare('_test_trades\\167.jpg')])

print(CATEGORIES[int(prediction[0][0])])

print('1'+CATEGORIES[int(prediction1[0][0])])
print('2'+CATEGORIES[int(prediction2[0][0])])
print('3'+CATEGORIES[int(prediction3[0][0])])
print('4'+CATEGORIES[int(prediction4[0][0])])
print('5'+CATEGORIES[int(prediction5[0][0])])
print('6'+CATEGORIES[int(prediction6[0][0])])
print('7'+CATEGORIES[int(prediction7[0][0])])
print('8'+CATEGORIES[int(prediction8[0][0])])
print('9'+CATEGORIES[int(prediction20[0][0])])
print('10'+CATEGORIES[int(prediction21[0][0])])
print('11'+CATEGORIES[int(prediction22[0][0])])
print('12'+CATEGORIES[int(prediction9[0][0])])
print('13'+CATEGORIES[int(prediction10[0][0])])
print('14'+CATEGORIES[int(prediction11[0][0])])
print('15'+CATEGORIES[int(prediction12[0][0])])
print('16'+CATEGORIES[int(prediction13[0][0])])
print('17'+CATEGORIES[int(prediction14[0][0])])
print('18'+CATEGORIES[int(prediction15[0][0])])
print('19'+CATEGORIES[int(prediction16[0][0])])
print('20'+CATEGORIES[int(prediction17[0][0])])
print('21'+CATEGORIES[int(prediction18[0][0])])
print('22'+CATEGORIES[int(prediction19[0][0])])