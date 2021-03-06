import tensorflow as tf
from PIL import Image
import cv2
import time
import numpy as np
from yolo import YOLO

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def predict_image(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, _, _, _ = yolo.detect_image(image)
            r_image.show()


if __name__ == '__main__':
    yolo = YOLO()
    predict_image(yolo)
