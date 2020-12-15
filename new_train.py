import os

from gensim.downloader import base_dir
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
# 训练、验证数据集的目录
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.optimizers import RMSprop

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_NG_dir = os.path.join(train_dir, 'NG')


train_OK_dir = os.path.join(train_dir, 'OK')

validation_NG_dir = os.path.join(validation_dir, 'NG')


validation_OK_dir = os.path.join(validation_dir, 'OK')

print('total training NG images:', len(os.listdir(train_NG_dir)))
print('total training OK images:', len(os.listdir(train_OK_dir)))
print('total validation NG images:', len(os.listdir(validation_NG_dir)))
print('total validation OK images:', len(os.listdir(validation_OK_dir)))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4),metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,  # target directory
    target_size=(150, 150),  # resize图片
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

hist = model.fit(
    train_generator,
    steps_per_epoch=40,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50
)



model.save('1.h5')