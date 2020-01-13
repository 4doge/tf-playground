import cv2

LABEL = 'supper'
video = cv2.VideoCapture(f'videos/{LABEL}.mov')
currentframe = 0

while(True):
    ret, frame = video.read()
    if ret:
        name = f'train_art_imgs/{LABEL}/{LABEL}-{currentframe}.jpg'
        print(f'Creating - {name}')
        cv2.imwrite(name, frame)
        currentframe += 1
    else:
        break

video.release()
cv2.destroyAllWindows()
