import os
import time
from collections import OrderedDict
from utils.utils import get_dataset
from tqdm import tqdm
from PIL import Image
import numpy as np
from utils.data_utils import preprocess_image
from utils.utils import get_classes
from yolo_component.c_postprocess import yolo4_postprocess
from yolo import YOLO
from model import get_yolo4_inference_model


class mAP_YOLO(YOLO):
    def _generate_model(self):
        self.score = 0.01
        self.iou = 0.5
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

        out_classnames = [self.class_names[c] for c in out_classes]
        return out_boxes, out_classnames, out_scores


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
    os.makedirs('result', exist_ok=True)
    result_file = open(os.path.join('result', 'detection_result.txt'), 'w')

    pred_classes_records = OrderedDict()
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
        # print('Found {} boxes for {}'.format(len(pred_boxes), image_name))
        print(pred_boxes, pred_classes, pred_scores)

        pbar.update(1)

        # save prediction result to txt
        result_file.write(image_name)
        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            xmin, ymin, xmax, ymax = box
            box_annotation = " %d,%d,%d,%d,%d,%f" % (
                int(xmin), int(ymin), int(xmax), int(ymax),int(cls), score)
            result_file.write(box_annotation)
        result_file.write('\n')
        result_file.flush()

        # if save_result:
        #
        #     gt_boxes, gt_classes, gt_scores = transform_gt_record(gt_records, class_names)
        #
        #     result_dir = os.path.join('result', 'detection')
        #     os.makedirs(result_dir, exist_ok=True)
        #     colors = get_colors(class_names)
        #     image_array = draw_boxes(image_array, gt_boxes, gt_classes, gt_scores, class_names, colors=None,
        #                              show_score=False)
        #     image_array = draw_boxes(image_array, pred_boxes, pred_classes, pred_scores, class_names, colors)
        #     image = Image.fromarray(image_array)
        #     # here we handle the RGBA image
        #     if (len(image.split()) == 4):
        #         r, g, b, a = image.split()
        #         image = Image.merge("RGB", (r, g, b))
        #     image.save(os.path.join(result_dir, image_name.split(os.path.sep)[-1]))

        # Nothing detected
        if pred_boxes is None or len(pred_boxes) == 0:
            continue

        for box, cls, score in zip(pred_boxes, pred_classes, pred_scores):
            pred_class_name = class_names[cls]
            xmin, ymin, xmax, ymax = box
            coordinate = "{},{},{},{}".format(xmin, ymin, xmax, ymax)

            # append or add predict class item
            record = [os.path.basename(image_name), coordinate, score]
            if pred_class_name in pred_classes_records:
                pred_classes_records[pred_class_name].append(record)
            else:
                pred_classes_records[pred_class_name] = list([record])

    # sort pred_classes_records for each class according to score
    for pred_class_list in pred_classes_records.values():
        pred_class_list.sort(key=lambda ele: ele[2], reverse=True)

    pbar.close()
    result_file.close()
    return pred_classes_records


annotation_file = '../configs/annotation_file/2007_test.txt'
class_names = get_classes('../configs/voc_classes.txt')
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
annotation_records, gt_classes_records = annotation_parse(annotation_file, class_names)

a = get_prediction_class_records(annotation_records, class_names, (416,416))
