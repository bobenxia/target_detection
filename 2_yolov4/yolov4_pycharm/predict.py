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


def predict_vedio(yolo, video_path='0'):
    capture = cv2.VideoCapture(0 if video_path == '0' else video_path)
    if not capture.isOpened():
        raise IOError("Couldn't open webcam or video")
    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref, frame = capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        # 进行检测
        image, _, _, _ = yolo.detect_image(frame)
        frame = np.array(image)
        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))

        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break

if __name__ == '__main__':
    yolo = YOLO()
    # predict_vedio(yolo)
    predict_image(yolo)