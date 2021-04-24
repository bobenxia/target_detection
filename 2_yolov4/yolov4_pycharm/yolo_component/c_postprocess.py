import tensorflow.keras.backend as K
import tensorflow as tf

from yolo_component.z_decode import batched_yolo4_boxes_and_scores
from yolo_component.z_decode import yolo4_boxes_and_scores


def batch_yolo4_postprocess(args,
                            anchors,
                            num_classes,
                            max_boxes=100,
                            score_threshold=0.1,
                            iou_threshold=0.4,
                            elim_grid_sense=False):
    """Postprocess for YOLOv4 model on gived input and return filtered boxes. """

    num_layers = len(anchors) // 3  # 特征图数量
    yolo_outputs = args[:num_layers]
    image_shape = args[num_layers]

    # 是否使用消除grid_sense
    if num_layers == 3:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    # 获取 input_shape，为了获取预测框在图片中的位置
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    batch_size = K.shape(image_shape)[0]
    boxes = []
    box_scores = []

    # 对每个特征图做处理，获得 boxes、 scores
    for l in range(num_layers):
        _boxes, _box_scores = batched_yolo4_boxes_and_scores(yolo_outputs[l],
                    anchors[anchor_mask[l]], num_classes, input_shape,
                    image_shape, scale_x_y=scale_x_y[l])
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 将每个特征图结果进行堆叠
    boxes = K.concatenate(boxes, axis=1)
    box_scores = K.concatenate(box_scores, axis=1)

    # 判断得分是否大于 score_threshold
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    def single_image_nms(b, batch_boxes, batch_scores, batch_classes):
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(num_classes):
            class_boxes = tf.boolean_mask(boxes[b], mask[b, :, c])
            class_box_scores = tf.boolean_mask(box_scores[b, :, c], mask[b, :, c])
            # 非极大抑制
            nms_index = tf.image.non_max_suppression(
                class_boxes,
                class_box_scores,
                max_boxes_tensor,
                iou_threshold=iou_threshold)
            # 获取非极大抑制后的结果
            class_boxes = K.gather(class_boxes, nms_index)
            class_box_scores = K.gather(class_box_scores, nms_index)
            classes = K.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = K.concatenate(boxes_, axis=0)
        scores_ = K.concatenate(scores_, axis=0)
        classes_ = K.concatenate(classes_, axis=0)

        batch_boxes = batch_boxes.write(b, boxes_)
        batch_scores = batch_scores.write(b, scores_)
        batch_classes = batch_classes.write(b, classes_)

        return b + 1, batch_boxes, batch_scores, batch_classes

    batch_boxes = tf.TensorArray(K.dtype(boxes), size=1, dynamic_size=True)
    batch_scores = tf.TensorArray(K.dtype(box_scores), size=1, dynamic_size=True)
    batch_classes = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    _, batch_boxes, batch_scores, batch_classes = tf.while_loop(lambda b, *args: b < batch_size, single_image_nms,
                                                                [0, batch_boxes, batch_scores, batch_classes])

    batch_boxes = batch_boxes.stack()
    batch_scores = batch_scores.stack()
    batch_classes = batch_classes.stack()

    return batch_boxes, batch_scores, batch_classes


def yolo4_postprocess(yolo_outputs,
                      image_shape,
                      anchors,
                      num_classes,
                      max_boxes=100,
                      score_threshold=0.1,
                      iou_threshold=0.4,
                      elim_grid_sense=False):
    """Postprocess for YOLOv4 model on given input and return filtered boxes."""

    num_layers = len(anchors) // 3  # default setting
    yolo_outputs = yolo_outputs
    image_shape = image_shape
    import numpy as np
    print(np.array(yolo_outputs[0].reshape((-1,7))).shape)

    if num_layers == 3:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    # 获取 input_shape，为了获取预测框在图片中的位置
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32

    # print("yolo_outputs",yolo_outputs)
    boxes = []
    box_scores = []

    # 对每个特征图做处理，获得 boxes、 scores
    for l in range(num_layers):
        _boxes, _box_scores = yolo4_boxes_and_scores(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, image_shape,
                                                     scale_x_y=scale_x_y[l])
        boxes.append(_boxes)
        box_scores.append(_box_scores)

    # 将每个特征图结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    # 判断得分是否大于 score_threshold
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # 非极大抑制
        nms_index = tf.image.non_max_suppression(
            class_boxes,
            class_box_scores,
            max_boxes_tensor,
            iou_threshold=iou_threshold)
        # 获取非极大抑制后的结果
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)

    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_
