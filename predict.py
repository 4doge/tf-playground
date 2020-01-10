import tensorflow as tf
import numpy as np
import cv2

IMG_WIDTH = 256
IMG_HEIGHT = 256

model = tf.keras.models.load_model('dogs_and_cats_model.h5')
model.summary()

img_cat = cv2.imread('cat_small.jpg')
img_cat = (np.expand_dims(img_cat, 0))
print(img_cat)
prediction1 = model.predict(img_cat)
print(prediction1)

img_dog = cv2.imread('dog_small.jpg')
img_dog = (np.expand_dims(img_dog, 0))
print(img_dog)
prediction2 = model.predict(img_dog)
print(prediction2)