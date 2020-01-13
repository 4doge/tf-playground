import os
import shutil

IMGS_DIR = 'train_art_imgs'
DEST_DIR = 'validation_art_imgs'

print(os.listdir(IMGS_DIR))

for folder in os.listdir(IMGS_DIR):
  for index, f in enumerate(os.listdir(f'{IMGS_DIR}/{folder}')):
    if index % 4 == 0:
      shutil.copyfile(f'{IMGS_DIR}/{folder}/{f}', f'{DEST_DIR}/{folder}/{f}')
      os.remove(f'{IMGS_DIR}/{folder}/{f}')
