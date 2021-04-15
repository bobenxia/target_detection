import tensorflow.keras.backend as K
import tensorflow as tf
import math

from yolo_component.z_decode import yolo4_decode


def _smooth_labels(y_true, label_smoothing):
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx)
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing


def sigmoid_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Compute sigmoid focal loss.
    Reference Paper:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002

    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
        gamma: exponent of the modulating factor (1 - p_t) ^ gamma.
        alpha: optional alpha weighting factor to balance positives vs negatives.

    # Returns
        sigmoid_focal_loss: Sigmoid focal loss, tensor of shape (?, num_boxes).
    """
    # bce 作为基本损失函数
    sigmoid_loss = K.binary_crossentropy(y_true, y_pred, from_logits=True)

    pred_prob = tf.sigmoid(y_pred)
    # 加号左边的 `y_true*pred_prob` 表示 将 y_true中为1的槽位置为 pred_prob对应槽位的值
    # 加号右边的 `(1-y_true)*(1-pred_prob)` 表示 将 y_true中为0的槽位置为 (1-pred_prob)对应槽位的值
    # p_t表示样本属于true class的概率，它反映了模型对这个样本的识别能力。
    # 显然，对于 p_t 越大的样本，我们越要打压它对loss的贡献，所以乘以 gamma 次方
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    modulating_factor = tf.pow(1.0 - p_t, gamma)

    # 加号左边的 y_true*alpha 表示将 y_true中等于1的槽位置为标量 alpha
    # 加号右边的 (ones-y_true)*(1-alpha) 则是将等于0的槽位置为 1-alpha
    alpha_weight_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    loss = alpha_weight_factor * modulating_factor * sigmoid_loss

    return loss


def softmax_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.maximum(tf.minimum(y_pred, 1 - 1e-15), 1e-15)

    # Calculate Cross Entropy
    cross_entropy = -y_true * tf.math.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy

    return loss


def box_iou(b1, b2):
    """
    Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxs = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxs = K.minimum(b1_maxs, b2_maxes)
    intersect_wh = K.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[0] * intersect_wh[1]

    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]

    union_area = b1_area + b2_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    return iou


def box_diou(b_true, b_pred, use_ciou=False):
    """
    Calculate DIoU/CIoU loss on anchor boxes
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Parameters
    ----------
    b_true: GT boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b_pred: predict boxes tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    use_ciou: bool flag to indicate whether to use CIoU loss type

    Returns
    -------
    diou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    b_true_xy = b_true[..., :2]
    b_true_wh = b_true[..., 2:4]
    b_true_wh_half = b_true_wh / 2.
    b_true_mins = b_true_xy - b_true_wh_half
    b_true_maxes = b_true_xy + b_true_wh_half

    b_pred_xy = b_pred[..., :2]
    b_pred_wh = b_pred[..., 2:4]
    b_pred_wh_half = b_pred_wh / 2.
    b_pred_mins = b_pred_xy - b_pred_wh_half
    b_pred_maxes = b_pred_xy + b_pred_wh_half

    intersect_mins = K.maximum(b_true_mins, b_pred_mins)
    intersect_maxes = K.minimum(b_true_maxes, b_pred_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    b_true_area = b_true_wh[..., 0] * b_true_wh[..., 1]
    b_pred_area = b_pred_wh[..., 0] * b_pred_wh[..., 1]

    union_area = b_true_area + b_pred_area - intersect_area
    # calculate IoU, add epsilon in denominator to avoid dividing by 0
    iou = intersect_area / (union_area + K.epsilon())

    # -------- 上面是计算普通 iou -----------------------------------

    # box center distance
    center_distance = K.sum(K.square(b_true_xy - b_pred_xy), axis=-1)
    # get enclosed area
    enclose_mins = K.minimum(b_true_mins, b_pred_mins)
    enclose_maxes = K.maximum(b_true_maxes, b_pred_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    # get enclosed diagonal distance
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal + K.epsilon())

    if use_ciou:
        # calculate param v and alpha to extend to CIoU
        v = 4 * K.square(tf.math.atan2(b_true_wh[..., 0], b_true_wh[..., 1]) - tf.math.atan2(b_pred_wh[..., 0],
                                                                                             b_pred_wh[..., 1])) / (
                    math.pi * math.pi)

        # a trick: here we add an non-gradient coefficient w^2+h^2 to v to customize it's back-propagate,
        #          to match related description for equation (12) in original paper
        #
        #
        #          v'/w' = (8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (h/(w^2+h^2))          (12)
        #          v'/h' = -(8/pi^2) * (arctan(wgt/hgt) - arctan(w/h)) * (w/(w^2+h^2))
        #
        #          The dominator w^2+h^2 is usually a small value for the cases
        #          h and w ranging in [0; 1], which is likely to yield gradient
        #          explosion. And thus in our implementation, the dominator
        #          w^2+h^2 is simply removed for stable convergence, by which
        #          the step size 1/(w^2+h^2) is replaced by 1 and the gradient direction
        #          is still consistent with Eqn. (12).

        v = v * tf.stop_gradient(b_pred_wh[..., 0] * b_pred_wh[..., 0] + b_pred_wh[..., 1] * b_pred_wh[..., 1])

        alpha = v / (1.0 - iou + v)
        diou = diou - alpha * v

    diou = K.expand_dims(diou, -1)
    return diou


def yolo4_loss(args, anchors, num_classes, ignore_thresh=.5, label_smoothing=0, elim_grid_sense=False,
              use_focal_loss=False, use_focal_obj_loss=False, use_softmax_loss=False, use_diou_loss=True):
    """
    YOLOv4 loss function.

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    """
    # ---------------------------------------------------------------------------------------------------#
    #   将预测结果和实际ground truth分开，args是[*model_body.output, *y_true]
    #   y_true是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一个列表，包含三个特征层，shape分别为(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    # ---------------------------------------------------------------------------------------------------#
    num_layers = len(anchors) // 3
    y_true = args[num_layers:]
    yolo_outputs = args[:num_layers]
    # -----------------------------------------------------------#
    #   13x13的特征层对应的anchor是[142, 110], [192, 243], [459, 401]
    #   26x26的特征层对应的anchor是[36, 75], [76, 55], [72, 146]
    #   52x52的特征层对应的anchor是[12, 16], [19, 36], [40, 28]
    # -----------------------------------------------------------#
    if num_layers == 3:
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.1, 1.2] if elim_grid_sense else [None, None, None]
    else:
        anchor_mask = [[3, 4, 5], [0, 1, 2]]
        scale_x_y = [1.05, 1.05] if elim_grid_sense else [None, None]

    # 得到input_shpae为416,416
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[i])[1:3], K.dtype(y_true[0])) for i in
                   range(num_layers)]  # grid_shape是指特征图shape
    loss = 0
    num_pos = 0
    total_location_loss = 0
    total_confidence_loss = 0
    total_class_loss = 0
    batch_size = K.shape(yolo_outputs[0])[0]  # batch size
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    # 依次计算特征图的损失值
    for i in range(num_layers):
        # 物体置信度和类别置信度
        object_mask = y_true[i][..., 4:5]
        true_class_probs = y_true[i][..., 5:]
        # 是否使用标签平滑
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)
            true_objectness_probs = _smooth_labels(object_mask, label_smoothing)
        else:
            true_objectness_probs = object_mask

        # 使用 yolo_decode 解码预测图，输出：(以 13*13 特征图举例)
        # - 网格 gird, 结构是(13, 13, 1, 2)，数值为0~12的全遍历二元组；
        # - 预测值 raw_pred:
        # - pred_xy和pred_wh都是归一化后的起始点xy和宽高wh，xy的结构是(?, 13, 13, 3, 2)，wh的结构是(?, 13, 13, 3, 2)；
        grid, raw_pred, pred_xy, pred_wh = yolo4_decode(yolo_outputs[i], anchors[anchor_mask[i]],
                                                        num_classes, input_shape, scale_x_y=scale_x_y[i],
                                                        calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        # - y_true的第0和1位是中心点xy的相对位置，范围是0~1；y_true的第2和3位是宽高wh的相对input_shape的位置，范围是0~1；
        # - raw_true_xy: 在网络中的中心点 xy, 偏移数据，值的范围是 0~1；
        # - raw_true_wh：在网络中的 wh 针对于 anchors 的比例，再转换为log形式，范围是有正有负；
        # - box_loss_scale：计算 wh 权重，取值范围（1~2）；
        raw_true_xy = y_true[i][..., :2] * grid_shapes[i][::-1] - grid
        raw_true_wh = K.log(y_true[i][..., 2:4] / anchors[anchor_mask[i] * input_shape] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))  # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

        # 根据ignore_thresh 生成，ignore_mask，将预测框pred_box和真值框true_box计算IoU，
        # 抑制不需要的anchor框的值，即IoU小于最大阈值的anchor框。
        # ignore_mask的shape是(?, ?, ?, 3, 1)，第0位是批次数，第1~2位是特征图尺寸。
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[i][b, ..., 0:4], object_mask_bool[b, ..., 0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh, K.dtype(true_box)))
            return b + 1, ignore_mask

        _, ignore_mask = tf.while_loop(lambda b, *args: b < batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # 第一部分损失：置信度的损失值
        # condidence_loss
        # 两部分组成，第一部分是存在物体的损失值，第 2 部分是不存物体的损失值，其中乘以掩码 ignore_mask,
        # 忽略预测框中 IOU大于阈值的框
        if use_focal_obj_loss:
            # Focal loss for objectness confidence
            confidence_loss = sigmoid_focal_loss(true_objectness_probs, raw_pred[..., 4:5])
        else:
            confidence_loss = object_mask * K.binary_crossentropy(true_objectness_probs, raw_pred[..., 4:5],
                                                                  from_logits=True) + \
                              (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                        from_logits=True) * ignore_mask

        # 第二部分损失：类别损失
        # class_loss
        if use_focal_loss:
            # Focal loss for classification score
            if use_softmax_loss:
                class_loss = softmax_focal_loss(true_class_probs, raw_pred[..., 5:])
            else:
                class_loss = sigmoid_focal_loss(true_class_probs, raw_pred[..., 5:])
        else:
            if use_softmax_loss:
                # use softmax style classification output
                class_loss = object_mask * K.expand_dims(
                    K.categorical_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True), axis=-1)
            else:
                # use sigmoid style classification output
                class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)

        # 第三部分损失：定位损失
        # location_loss
        if use_diou_loss:
            # Calculate DIoU loss as location loss
            raw_true_box = y_true[i][..., 0:4]
            diou = box_diou(raw_true_box, pred_box)
            diou_loss = object_mask * box_loss_scale * (1 - diou)
            diou_loss = K.sum(diou_loss) / batch_size_f
            location_loss = diou_loss
        else:
            # Standard YOLOv3 location loss
            # K.binary_crossentropy is helpful to avoid exp overflow.
            xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                           from_logits=True)
            wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
            xy_loss = K.sum(xy_loss) / batch_size_f
            wh_loss = K.sum(wh_loss) / batch_size_f
            location_loss = xy_loss + wh_loss

        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += location_loss + confidence_loss + class_loss
        total_location_loss += location_loss
        total_confidence_loss += confidence_loss
        total_class_loss += class_loss

    # Fit for tf 2.0.0 loss shape
    loss = K.expand_dims(loss, axis=-1)

    return loss, total_location_loss, total_confidence_loss, total_class_loss
