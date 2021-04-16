from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from yolo_component.c_postprocess import batch_yolo4_postprocess
from yolo_component.b_neck_and_body import yolo4_body
from yolo_component.z_loss import yolo4_loss
from utils.model_utils import add_metrics


def get_yolo4_model(num_anchors, num_classes, weights_path=None, input_tensor=None, input_shape=None):
    # prepare input tensor
    if input_shape:
        input_tensor = Input(shape=input_shape, name='image_input')

    if input_tensor is None:
        input_tensor = Input(shape=(None, None, 3), name='image_input')

    model_function = yolo4_body
    backbone_len = 250

    if weights_path:
        model_body = model_function(input_tensor, num_anchors // 3, num_classes, weights_path=weights_path)
    else:
        model_body = model_function(input_tensor, num_anchors // 3, num_classes)

    return model_body, backbone_len


def get_yolo4_inference_model(anchors, num_classes, weights_path=None,
                              input_shape=None, score_threshold=0.1, iou_threshold=0.4, elim_grid_sense=False):
    """create the inference model, for YOLOv4"""
    num_anchors = len(anchors)

    image_shape = Input(shape=(2,), dtype='int64', name='image_shape')
    model_body, _ = get_yolo4_model(num_anchors, num_classes, weights_path, input_shape=input_shape)

    if weights_path:
        model_body.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    boxes, scores, classes = Lambda(batch_yolo4_postprocess, name='yolo4_postprocess',
                                    arguments={
                                        'anchors': anchors,
                                        'num_classes': num_classes,
                                        'score_threshold': score_threshold,
                                        'iou_threshold': iou_threshold,
                                        'elim_grid_sense': elim_grid_sense
                                    })([*model_body.output, image_shape])
    model = Model([model_body.input, image_shape], [boxes, scores, classes])

    return model


def get_yolo4_train_model(anchors, num_classes, weights_path=None, freeze_level=1,
                          optimizer=Adam(lr=1e-3, decay=0), label_smoothing=0, elim_grid_sense=False):
    """Create the training model, for YOLOv4"""
    num_anchors = len(anchors)
    num_feature_layers = num_anchors // 3

    # feature map target value, so its shape should be like:
    # [
    #  (image_height/32, image_width/32, 3, num_classes+5),
    #  (image_height/16, image_width/16, 3, num_classes+5),
    #  (image_height/8, image_width/8, 3, num_classes+5)
    # ]
    y_true = [Input(shape=(None, None, 3, num_classes + 5), name='y_true_{}'.format(l)) for l in
              range(num_feature_layers)]

    model_body, backbone_len = get_yolo4_model(num_feature_layers, num_anchors, num_classes)
    print('Create model')

    if weights_path:
        model_body.load_weights(weights_path, by_name=True)

    if freeze_level in [1, 2]:
        # Freeze the backbone part or freeze all but final feature map & input layers
        num = (backbone_len, len(model_body.layers) - 3)[freeze_level - 1]  # 通过01选择前面或者后面的数
        for i in range(num):
            model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    elif freeze_level == 0:
        # unfreeze all layers.
        for i in range(len(model_body.layers)):
            model_body.layers[i].trainable = True
        print('Unfree all of the layers. ')

    model_loss, location_loss, confidence_loss, class_loss = Lambda(yolo4_loss, name='yolo_loss',
                                                                    arguments={
                                                                        'anchors': anchors,
                                                                        'num_classes': num_classes,
                                                                        'ignore_thresh': 0.5,
                                                                        'label_smoothing': label_smoothing,
                                                                        'elim_grid_sense': elim_grid_sense
                                                                    })([*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    loss_dict = {'location_loss': location_loss, 'confidence_loss': confidence_loss, 'class_loss': class_loss}
    add_metrics(model, loss_dict)

    model.compile(optimizer=optimizer, loss={
        # use custom yolo_loss Lambda layer.
        'yolo_loss': lambda y_true, y_pred: y_pred})

    return model
