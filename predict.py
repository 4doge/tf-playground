import tensorflow as tf
import numpy as np
import cv2
import pathlib
from utils import get_label_index

IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_DIR = pathlib.Path('train_imgs')
labels = [folder.name for folder in TRAIN_DIR.glob('*')]

model = tf.keras.models.load_model('dogs_and_cats_and_flowers.h5')
model.summary()

img = cv2.imread('dog.jpg')
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
img = img / 255.0


def callback(*args, **kwargs):
    print(args)
    print(kwargs)


prediction1 = model.predict(img)
class_index = model.predict_classes(img)
print(prediction1[0])
print(class_index)
CLASS_NAMES = np.array([folder.name for folder in TRAIN_DIR.glob('*')])
print(f'class is {list(CLASS_NAMES[class_index])}')
# print(f'Label is: {get_label_index(class_index, is_label=False)}')

if __name__ == '__main__':
    pass
