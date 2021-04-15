import tensorflow.keras.backend as K


def yolo4_decode(feats, anchors, num_classes, input_shape, scale_x_y=None, calc_loss=False):
    """Decode final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # ----------------------------------------------------------------------------------------------------------
    # 生成 grid 网格基准 (13, 13, 1, 2)
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    # Reshape to ([batch_size, height, width, num_anchors, (num_classes+5)])
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust predictions to each spatial grid point and anchor size.
    # box_xy 数值范围调整为【0-1】(归一化)
    # box_wh 数值范围调整为 【0-1】(归一化)，输入尺寸是使用backbone的最小特征图尺寸*stride得到的
    # 强调说明一下：这里 box_xy 是相对于grid 的位置（说成input似乎也行）；box_wh是相对于 input_shape大小
    # scale_x_y是一个 trick，见下文链接
    if scale_x_y:
        # Eliminate grid sensitivity trick involved in YOLOv4
        #
        # Reference Paper & code:
        #     "YOLOv4: Optimal Speed and Accuracy of Object Detection"
        #     https://arxiv.org/abs/2004.10934
        #     https://github.com/opencv/opencv/issues/17148
        #     https://zhuanlan.zhihu.com/p/139724869
        box_xy_tmp = K.sigmoid(feats[..., :2]) * scale_x_y - (scale_x_y - 1) / 2
        box_xy = (box_xy_tmp + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    else:
        box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[..., ::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[..., ::-1], K.dtype(feats))
    # sigmoid objectness scores 置信度解码
    box_confidence = K.sigmoid(feats[..., 4:5])
    # class probs 类别解码
    box_class_probs = K.sigmoid(feats[..., 5:])

    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo4_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """计算出预测框相对于原图像的位置和大小"""
    input_shape = K.cast(input_shape, K.dtype(box_xy))
    image_shape = K.cast(image_shape, K.dtype(box_xy))

    # reshape the image_shape tensor to align with boxes dimension
    image_shape = K.reshape(image_shape, [-1, 1, 1, 1, 2])

    new_shape = K.round(image_shape * K.min(input_shape / image_shape))  # （416，312）

    #  这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
    offset = (input_shape - new_shape) / 2. / input_shape  # （0，0.125）
    scale = input_shape / new_shape  # (1, 1.333)

    # 举例，在现在还未进行任何尺度变换，假设有一个坐标为（0.4，0.3），wh为（0.2，0.2）的框。
    # 上面这个数值的意义为：这个坐标是在input（416，416） 的相对位置和相对宽高，实际上，input
    # 的（:,0:0.125）的区域都没有意义，（:,0.875:1）也是。
    #
    # 现在我们需要把上面这个相对参照物换成 image，而image经过缩放后的区域是（416，312）。上面的
    # offset 和 Scale 就是为了完成一个参照物替换的
    box_xy_image = (box_xy - offset) * scale
    box_wh_image = box_wh * scale
    # box_xy_image 和 box_wh_image 就是以image 为参考系的相对位置和相对宽高

    box_mins = box_xy_image - (box_wh_image / 2.)
    box_maxes = box_xy_image + (box_wh_image / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # x_min
        box_mins[..., 1:2],  # y_min
        box_maxes[..., 0:1],  # x_max
        box_maxes[..., 1:2]  # y_max
    ])

    # Scale boxes back to original image shape.
    # 通过乘积获得了真实宽高和位置了
    image_wh = image_shape[..., ::-1]
    boxes *= K.concatenate([image_wh, image_wh])
    return boxes


def batched_yolo4_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, scale_x_y):
    """Process Conv layer output"""
    box_xy, box_wh, box_confidence, box_class_probs = yolo4_decode(feats,
                        anchors, num_classes, input_shape, scale_x_y=scale_x_y)

    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[1:3]  # height, width
    total_anchor_num = grid_shape[0] * grid_shape[1] * num_anchors

    boxes = yolo4_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, total_anchor_num, 4])

    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, total_anchor_num, num_classes])

    return boxes, box_scores