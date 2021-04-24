import os
import time

import numpy as np
from PIL import Image
from tensorflow.keras import backend as K

from model import get_yolo4_inference_model
from utils.data_utils import preprocess_image
from utils.utils import get_colors, get_classes, get_anchors, draw_boxes
from tensorflow.keras.layers import Input, Lambda

import pycuda.driver as cuda
import tensorrt as trt

# '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/
# r'C:\Users\xia\Documents\codes\20210403_目标检测\target_detection\2_yolov4\yolov4_pycharm\
from yolo_component.c_postprocess import batch_yolo4_postprocess, yolo4_postprocess
import ctypes

try:
    ctypes.cdll.LoadLibrary('./plugins/libyolo_layer.so')
except OSError as e:
    raise SystemExit('ERROR: failed to load ./plugins/libyolo_layer.so.  '
                     'Did you forget to do a "make" in the "./plugins/" '
                     'subdirectory?') from e

default_config = {
    "anchors_path": '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/configs/yolo_anchors.txt',
    "classes_path": '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/configs/coco_classes.txt',
    "score": 0.5,
    "iou": 0.4,
    "model_image_size": (416, 416),
    "elim_grid_sense": False,
}


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """
    Allocate all host/device in/out buffers required for an engine.
    :param engine:
    :return:
    """
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    assert 3 <= len(engine) <= 4  # expect 1 input, plus 2 or 3 outputs

    for binding in engine:
        bindings_dims = engine.get_binding_shape(binding)
        if len(bindings_dims) == 4:
            # explicit batch case (TensorRT 7+)
            size = trt.volume(bindings_dims)
        elif len(bindings_dims) == 3:
            # implicit batch case (TensorRT 6 or older)
            size = trt.volume(bindings_dims) * engine.max_batch_size
        else:
            raise ValueError('bad dims of binding %s: %s' % (binding, str(bindings_dims)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        # Append the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            assert size % 7 == 0
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1

    return inputs, outputs, bindings, stream


def inference_in_cuda(context, bindings, inputs, outputs, stream, batch_size=1):
    """Do inference (for TensorRT 6.x or lower)

    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def inference_in_cuda_v2(context, bindings, inputs, outputs, stream):
    """Do inference (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtYOLO(object):
    _defaults = default_config

    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，
    # 但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等。
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name' " + n + " ' "

    def __init__(self, model_name, cuda_context):
        super(TrtYOLO, self).__init__()
        self.__dict__.update(self._defaults)  # 还能这样？
        self.class_names = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)

        self.model_name = model_name
        self.cuda_context = cuda_context
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        self.inference_method = inference_in_cuda if trt.__version__[0] < '7' else inference_in_cuda_v2

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resource') from e

    def __del__(self):
        """Free CUDA memories"""
        del self.outputs
        del self.inputs
        del self.stream

    def _load_engine(self):
        trt_model = '/home/xia/Documents/1_code/16_target_detection/target_detection/2_yolov4/yolov4_pycharm/weights/%s.trt' % self.model_name
        with open(trt_model, 'rb') as f, \
                trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def predict(self):
        # 输出三张特征图
        trt_outputs = self.inference_method(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream
        )
        print(np.array(trt_outputs[0]).shape)
        print(np.array(trt_outputs[1]).shape)
        print(np.array(trt_outputs[2]).shape)
        print(np.array(trt_outputs[0].reshape((-1,7)))[7][6])
        image_shape = Input(shape=(2,), dtype='int64', name='image_shape')
        out_boxes, out_scores, out_classes = yolo4_postprocess(yolo_outputs=trt_outputs,
                                                               image_shape=image_shape,
                                                               anchors=self.anchors,
                                                               num_classes=self.num_classes,
                                                               score_threshold=self.score,
                                                               iou_threshold=self.iou,
                                                               elim_grid_sense=self.elim_grid_sense)
        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        image_data = preprocess_image(image, self.model_image_size)
        image_data = image_data[1:]  # 去掉 preprocess_image 中添加的 batch 维

        self.inputs[0].host = np.ascontiguousarray(image_data)  # 一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict()
        end = time.time()

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        # draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores
