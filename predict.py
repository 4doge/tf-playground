import tensorflow as tf
import numpy as np
import cv2
import pathlib
import matplotlib.pyplot as plt

IMG_WIDTH = 256
IMG_HEIGHT = 256
TRAIN_DIR = pathlib.Path('train_art_imgs')
labels = [folder.name for folder in TRAIN_DIR.glob('*')]

model = tf.keras.models.load_model('art_model.h5')
model.summary()

img = cv2.imread('mona_clear.jpg')
img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
# cv2.imshow('Image', img)
# cv2.waitKey(0)
img = np.reshape(img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
img = img / 255.0
prediction1 = model.predict(img)
print(prediction1[0])

predicted_label = labels[np.argmax(prediction1[0])]
print(predicted_label)
# index = prediction1.argmax(axis=-1)[0]
# predicted_label = sorted(labels)[index]

# print(predicted_label)

# def plot_value_array(i, predictions_array, true_label):
#   predictions_array, true_label = predictions_array, true_label[i]
#   plt.grid(False)
#   plt.xticks(range(7))
#   plt.yticks([])
#   thisplot = plt.bar(range(7), predictions_array, color="#777777")
#   plt.ylim([0, 1])
#   predicted_label = np.argmax(predictions_array)

#   thisplot[predicted_label].set_color('red')
#   thisplot[true_label].set_color('blue')

# plot_value_array(1, prediction1[0], range(7))
# plt.xticks(range(7), labels, rotation=45)
# plt.show()
