import tensorflow as tf
import numpy as np
import cv2
import pathlib

IMG_WIDTH = 224
IMG_HEIGHT = 224
TRAIN_DIR = pathlib.Path('train_art_imgs')

model = tf.keras.models.load_model('keras_model.h5')
labels = [folder.name for folder in TRAIN_DIR.glob('*')]
print(labels)

img = cv2.imread('validation_art_imgs/oops/oops-3.jpg')
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, 3])

predictions = model.predict(img)[0].tolist()
print(predictions)
for i, p in enumerate(predictions):
    print(p)
    name = labels[i]
    print('Prediction: {} ({:4.1f}%)'.format(name, 100*p))
