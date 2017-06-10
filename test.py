import os
import sys
import keras
from keras.models import model_from_json
from keras.models import load_model
from keras.models import Sequential
import numpy as np 
from keras.preprocessing import image

input_dir = './data/train'
categories = [name for name in sorted(os.listdir(input_dir)) if name != ".DS_Store"]

if len(sys.argv) != 2:
    print("usage: python first_chk.py [filename]")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

img_height, img_width = 150, 150
model = Sequential()

model = model_from_json(open('train.json').read())

model.load_weights('train.h5')

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()


img = image.load_img(filename, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = x / 255.0

pred = model.predict(x)
print(pred)

for pre in pred:
  y = pre.argmax()
  print("Output: ", y, categories[y])

