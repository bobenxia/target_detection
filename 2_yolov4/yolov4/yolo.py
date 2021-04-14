import os
import time
from tensorflow.keras.layers import Input, Lambda

default_config = {
        "model_type": 'tiny_yolo3_darknet',
        "weights_path": os.path.join('weights', 'yolov3-tiny.h5'),
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'tiny_yolo3_anchors.txt'),
        "classes_path": os.path.join('configs', 'coco_classes.txt'),
        "score" : 0.1,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "elim_grid_sense": False,
        "gpu_num" : 1,
    }


def get_yolo3_inference_model(model_type, anchors, num_classes, weights_path=None,
                              input_shape=None, confidence=0.1, iou_threshold=0.4, elim_grid_sense=False):
    """create the inference model, for YOLOv3"""
    num_anchors = len(anchors)
    num_feature_layers = num_anchors // 3

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')
    # TODO: get_yolo3_mode
    model_body, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=input_shape)

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)
        print('Load weight {}'.format(weights_path))

    # TODO: batch_yolo_postprocess
    boxes, scores, classes = Lambda(batch_yolo_postprocess, name='yolo3_postprocess',
                                    arguments={
                                        'anchors': anchors,
                                        'num_classes': num_classes,
                                        'confidence': confidence,
                                        'iou_threshold': iou_threshold,
                                        'elim_grid_sense': elim_grid_sense
                                    })([*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model

