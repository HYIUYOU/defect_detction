from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
import numpy as np

vgg16_model = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2,activation='softmax'))

model = Sequential()
model.add(vgg16_model)
model.add(top_model)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)
batch_size = 32

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150,150),
    batch_size=batch_size
)

test_generator = test_datagen.flow_from_directory(
    'test',
    target_size=(150,150),
    batch_size=batch_size
)
print(train_generator.class_indices)

model.compile(optimizer=SGD(lr=1e-4,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_generator,epochs=2,validation_data=test_generator)

model.save('model_vgg16.h5')