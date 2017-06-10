'''
"Building powerful image classification models using very little data"
by Yoshi Sakai as Bluemooninc. https://github.com/bluemooninc
It uses data that can be downloaded at:
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created apple/ pineapple and pen subfolders inside train/ and validation/
- put the apple,pineapple,pen pictures index 1-24 in data/train/ each folder
- put the dpple,pineapple,pen pictures index 25-30 in data/validation/ each folder
So that we have 24 training examples for each class, and 6 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        apple/
            apple001.jpg
            apple002.jpg
            ...
        pineapple/
            pineapple001.jpg
            pineapple002.jpg
            ...
        pen/
            pen001.jpg
            pen002.jpg
            ...
    validation/
        apple/
            apple025.jpg
            apple026.jpg
            ...
        pineapple/
            pineapple025.jpg
            pineapple026.jpg
            ...
        pen/
            pen025.jpg
            pen026.jpg
            ...
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = './data/train'
validation_data_dir = './data/validation'
nb_train_samples = 72
nb_validation_samples = 18
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3)) # class number of images
model.add(Activation('softmax'))
## 1.binary_crossentropy
## 2.categorical_crossentropy
## 3.sparse_categorical_crossentropy
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


model_json_str = model.to_json()
open('train.json', 'w').write(model_json_str)
model.save_weights('train.h5')

