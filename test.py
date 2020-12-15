from keras.models import load_model
import numpy as np
from keras_preprocessing.image import load_img, img_to_array

label = np.array(['NG','Ok'])

model = load_model('model_vgg16.h5')

image = load_img('testImg/2.jpg')

image = image.resize((150,150))
image = img_to_array(image)
image = np.expand_dims(image,0)
print(label[model.predict_classes(image)])