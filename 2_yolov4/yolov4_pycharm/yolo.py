import os
import time

import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from model import get_yolo4_inference_model
from utils.data_utils import preprocess_image
from utils.utils import get_colors, get_classes, get_anchors, draw_boxes

# '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4/weights/yolov4.h5',
default_config = {
    "weights_path": '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/weights/yolov4.h5',
    "anchors_path": '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/configs/yolo_anchors.txt',
    "classes_path": '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/configs/coco_classes.txt',
    "score": 0.5,
    "iou": 0.4,
    "model_image_size": (416, 416),
    "elim_grid_sense": False,
}


class YOLO(object):
    _defaults = default_config

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，
    # 但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name' " + n + " ' "

    def __init__(self, **kwargs):
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults)  # 还能这样？
        self.__dict__.update(kwargs)  # update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.inference_model = self._generate_model()

    def _generate_model(self):
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # 先验框数量和种类数量
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # 载入模型
        inference_model = get_yolo4_inference_model(self.anchors, num_classes, weights_path=weights_path,
                                                    input_shape=self.model_image_size + (3,),
                                                    score_threshold=self.score,
                                                    iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)

        #
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        # draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores
