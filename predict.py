import tensorflow as tf
import numpy as np
import cv2
import pathlib

IMG_WIDTH = 128
IMG_HEIGHT = 128
TRAIN_DIR = pathlib.Path('train_imgs')
LABELS = [folder.name for folder in TRAIN_DIR.glob('*')]
print(LABELS)
model = tf.keras.models.load_model('dogs_and_cats_and_flowers2.h5')
model.summary()

img = cv2.imread('cat2.jpg')
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
img = img / 255.0

prediction1 = model.predict(img)
class_index = model.predict_classes(img)
print(prediction1[0])
print(class_index)
CLASS_NAMES = np.array([folder.name for folder in TRAIN_DIR.glob('*')])
print(f'Label  is {LABELS[class_index[0]]}')
# print(f'Label is: {get_label_index(class_index, is_label=False)}')

if __name__ == '__main__':
    pass
