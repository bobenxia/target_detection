from tensorflow.keras.layers import UpSampling2D, Concatenate, ZeroPadding2D
from tensorflow.keras.models import Model

from yolo_component.a_darknet import csp_darknet53_body
from yolo_component.z_layers import make_three_darknet_CBL, make_five_darknet_CBL
from yolo_component.z_layers import spp, compose, darknet_CBL, darknet_Conv2D


def yolov4_neck(feature_maps, feature_channel_nums, num_anchors, num_classes):
    """
    """
    f1, f2, f3 = feature_maps # (f1:19x19|f2:38*38|f3:76*76 for input input)
    f1_channel_num, f2_channel_num, f3_channel_num = feature_channel_nums

    # feature map 1(19x19 for 608 input)
    x1 = make_three_darknet_CBL(f1, f1_channel_num//2)
    x1 = spp(x1)
    x1 = make_three_darknet_CBL(x1, f1_channel_num//2)
    x1_upsample = compose(
        darknet_CBL(f2_channel_num//2, (1, 1)),
        UpSampling2D(2))(x1)

    x2 = darknet_CBL(f2_channel_num//2, (1, 1))(f2)
    x2 = Concatenate()([x2, x1_upsample])

    # After concatenate: feature map 2(38x38 for 608 input)
    x2 = make_five_darknet_CBL(x2, f2_channel_num//2)
    x2_upsample = compose(
        darknet_CBL(f3_channel_num//2, (1, 1)),
        UpSampling2D(2))(x2)

    x3 = darknet_CBL(f3_channel_num//2, (1, 1))(f3)
    x3 = Concatenate()([x3, x2_upsample])

    # After concatenate: feature map 3(76x76 for 608 input)
    x3 = make_five_darknet_CBL(x3, f3_channel_num//2)

    # ----------------------------------------------------------------------------

    # output (76x76 for 608 input)
    y3 = compose(
        darknet_CBL(f3_channel_num//2, (1, 1)),
        darknet_Conv2D(num_anchors *(num_classes + 5), (1, 1), name='predict_conv_3'))(x3)

    # downsample fpn merge for feature map 3 & 2
    x3_downsample = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        darknet_CBL(f2_channel_num//2, (3, 3), strides=(2, 2)))(x3)
    x2 = Concatenate()([x3_downsample, x2])
    x2 = make_five_darknet_CBL(x2, f2_channel_num//2)

    # output (38x38 for 608 input)
    y2 = compose(
        darknet_CBL(f2_channel_num//2, (1, 1)),
        darknet_Conv2D(num_anchors *(num_classes + 5), (1, 1), name='predict_conv_2'))(x2)

    # downsample fpn merge for feature map 2 & 1
    x2_downsample = compose(
        ZeroPadding2D(((1, 0), (1, 0))),
        darknet_CBL(f1_channel_num//2, (3, 3), strides=(2, 2)))(x2)
    x1 = Concatenate()([x2_downsample, x1])
    x1 = make_five_darknet_CBL(x1, f1_channel_num//2)

    # output (19x19 for 608 input)
    y1 = compose(
        darknet_CBL(f1_channel_num, (3, 3)),
        darknet_Conv2D(num_anchors *(num_classes + 5), (1, 1), name='predict_conv_1'))(x1)

    return y1, y2, y3


def yolo4_body(inputs, num_anchors, num_classes, weights_path=None):
    """Create YOLOv4 model' backbone CNN body in keras"""
    darknet = Model(inputs, csp_darknet53_body(inputs))
    if weights_path is not None:
        darknet.load_weights(weights_path, by_name=True)

    f1 = darknet.output
    f2 = darknet.layers[204].output
    f3 = darknet.layers[131].output

    y1, y2, y3 = yolov4_neck((f1, f2, f3), (1024, 512, 256), num_anchors, num_classes)

    return Model(inputs, [y1, y2, y3])
