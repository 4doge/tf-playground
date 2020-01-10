import tensorflow as tf
import numpy as np
import cv2
import pathlib

IMG_WIDTH = 256
IMG_HEIGHT = 256
TRAIN_DIR = pathlib.Path('train_imgs')
labels = [folder.name for folder in TRAIN_DIR.glob('*')]

model = tf.keras.models.load_model('dogs_and_cats_model.h5')
model.summary()

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
img = img / 255.0
prediction1 = model.predict(img)
print(prediction1[0])
index = prediction1.argmax(axis=-1)[0]
predicted_label = sorted(labels)[index]
print(predicted_label)
