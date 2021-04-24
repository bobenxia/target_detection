from random import random

import tensorflow as tf
from PIL import Image
import cv2
import time
import numpy as np
from yolo_trt import TrtYOLO
from queue import Queue
from threading import Thread

import pycuda.autoinit  # This is needed for initializing CUDA driver


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def video_capture(frame_queue, yolo_image_queue):
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        # 格式转变，BGRtoRGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame_to_detect = Image.fromarray(np.uint8(frame_rgb))

        frame_queue.put(frame_rgb)
        yolo_image_queue.put(frame_to_detect)
    capture.release()


def inference(yolo_image_queue, detections_queue, fps_queue):
    while capture.isOpened():
        image = yolo_image_queue.get()
        t1 = time.time()
        detections_image, _, _, _ = yolo.detect_image(image)
        detections_queue.put(detections_image)
        fps = int(1 / (time.time() - t1))
        fps_queue.put(fps)
    capture.release()


def drawing(frame_queue, detections_queue, fps_queue):
    while capture.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if detections_queue is not None:
            detections_image = np.array(detections)
            image = cv2.cvtColor(detections_image, cv2.COLOR_RGB2BGR)
            image = cv2.putText(image, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Inference', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    frame_queue = Queue()
    yolo_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    yolo = TrtYOLO()
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise IOError("Couldn't open webcam or video")

    Thread(target=video_capture, args=(frame_queue, yolo_image_queue)).start()
    Thread(target=inference, args=(yolo_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()







