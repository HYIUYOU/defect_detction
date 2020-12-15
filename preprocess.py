from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(
    rescale=1/255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #rotation_range=40,
    #width_shift_range=0.3,
    # height_shift_range=0.3,
     horizontal_flip=True,
     fill_mode='nearest'
)
img = load_img('/Users/heyiyuan/Desktop/defect_detection/data/1.jpg')
x = img_to_array(img)
print(x.shape)

x = np.expand_dims(x,0)
print(x.shape)
i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='temp',save_prefix='new',save_format='jpeg'):
    i+=1
    if i == 1:
        break


