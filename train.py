import tensorflow as tf
import numpy as np
import pathlib
import os

BATCH_SIZE = 32
EPOCHS = 5
IMG_WIDTH = 256
IMG_HEIGHT = 256
AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_DIR = pathlib.Path('train_art_imgs')
VAL_DIR = pathlib.Path('validation_art_imgs')

train_image_count = len(list(TRAIN_DIR.glob('*/*.jpg')))
validation_image_count = len(list(VAL_DIR.glob('*/*.jpg')))
print(f'TRAIN IMAGES - {train_image_count}')
print(f'VALIDATION IMAGES - {validation_image_count}')

CLASS_NAMES = np.array([folder.name for folder in TRAIN_DIR.glob('*')])
print(f'CLASSES - {CLASS_NAMES}')

train_list_ds = tf.data.Dataset.list_files(str(TRAIN_DIR/'*/*'))
validation_list_ds = tf.data.Dataset.list_files(str(VAL_DIR/'*/*'))


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(file_path):
    label = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
validation_labeled_ds = validation_list_ds.map(
    process_path, num_parallel_calls=AUTOTUNE)


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = prepare_for_training(train_labeled_ds)
validation_ds = prepare_for_training(validation_labeled_ds)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu',
                           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(list(CLASS_NAMES)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(x=train_ds,
          epochs=EPOCHS,
          steps_per_epoch=train_image_count // BATCH_SIZE,
          validation_data=validation_ds,
          validation_steps=validation_image_count // BATCH_SIZE)

model.save('model.h5')
