import os
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from utils.data_utils import preprocess_image
from utils.utils import get_classes
from utils.utils import get_dataset
from yolo import YOLO

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class mAP_YOLO(YOLO):
    def __init__(self):
        super(mAP_YOLO, self).__init__()
        self.score = 0.01
        self.iou = 0.5

    def detect_image(self, image):
        print(self.score)
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)

        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)

        return out_boxes, out_classes, out_scores


def annotation_parse(annotation_file, class_names):
    """
    parse annotation file to get image dict and ground truth class dict

    Args:
        annotation_file: test annotation txt file
        class_names: list of class names

    Return:
        image dict would be like:
        annotation_records = {
            '/path/to/000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
            ...
        }
        ground truth class dict would be like:
        classes_records = {
            'car': [
                    ['000001.jpg','100,120,200,235'],
                    ['000002.jpg','85,63,156,128'],
                    ...
                   ],
            ...
        }
    """
    annotation_records = OrderedDict()
    classes_records = OrderedDict({class_name: [] for class_name in class_names})

    annotation_lines = get_dataset(annotation_file, shuffle=False)
    # annotation_lines would be like:
    # ['/path/to/000001.jpg 100,120,200,235,11 85,63,156,128,14',
    # ...,
    # ]
    for line in annotation_lines:
        box_records = {}
        image_name = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            # strip box coordinate and class
            class_name = class_names[int(box.split(',')[-1])]
            coordinate = ','.join(box.split(',')[:-1])
            box_records[coordinate] = class_name
            # append or add ground truth class item
            record = [os.path.basename(image_name), coordinate]
            if class_name in classes_records:
                classes_records[class_name].append(record)
            else:
                classes_records[class_name] = list([record])
        annotation_records[image_name] = box_records

    return annotation_records, classes_records


def get_prediction_class_records(annotation_records, class_names, model_image_size):
    """
    Do the predict with YOLO model on annotation images to get predict class dict

    Reutrn：
        predict class dict would contain image_name, coordinary and score, and
        sorted by score:
        pred_classes_records = {
            'car': [
                    ['000001.jpg','94,115,203,232',0.98],
                    ['000002.jpg','82,64,154,128',0.93],
                    ...
                   ],
            ...
        }
    """
    # create txt file to save prediction result, with
    # save format as annotation file but adding score, like:
    #
    # path/to/img1.jpg 50,100,150,200,0,0.86 30,50,200,120,3,0.95
    #
    result_file = open('../../outputs/detection_result.txt', 'w')

    pbar = tqdm(total=len(annotation_records), desc='Eval model')
    # annotation_records = {
    #             '/path/to/000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
    #             ...
    #         }
    yolo = mAP_YOLO()
    for (image_name, gt_records) in annotation_records.items():
        image = Image.open(image_name)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pred_boxes, pred_classes, pred_scores = yolo.detect_image(image)
        pbar.update(1)

        # save prediction result to txt
        result_file.write(image_name)
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            pred_class_name = class_names[cls]
            xmin, ymin, xmax, ymax = box
            box_annotation = " %d,%d,%d,%d,%d,%f" % (
                int(xmin), int(ymin), int(xmax), int(ymax), cls, score)
            result_file.write(box_annotation)
        result_file.write('\n')
        result_file.flush()

    pbar.close()
    result_file.close()


if __name__ == '__main__':
    annotation_file = '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/outputs/val2017.txt'
    class_names = get_classes('/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/configs/coco_classes.txt')
    print(class_names)

    annotation_records, gt_classes_records = annotation_parse(annotation_file, class_names)

    # print(annotation_records)
    get_prediction_class_records(annotation_records, class_names, (416,416))
