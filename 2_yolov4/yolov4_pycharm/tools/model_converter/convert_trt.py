import os, argparse
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description='convert model to trt')
parser.add_argument('--weights', type=str, required=False, help='path to weights file', default='../../weights/yolov4.h5')
parser.add_argument('--output_path', type=str, required=False, help='output path for generated trt model', default='../../outputs/trt_model')
parser.add_argument('--quantize_mode', type=str, required=False, help='quantize mode (int8, float16, float32)', default='float16')
parser.add_argument('--dataset', type=str, required=False, help='the path of dataset for optimization', default='')
parser.add_argument('loop',type=int, required=False, )
args = parser.parse_args()


def save_trt():
    assert args.quantize_model in ['int8', 'float16', 'float32'], "quantize_mode should be one of ['int8', 'float16', 'float32']"
    if args.quantize_model == 'int8':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.INT8,
            max_workspace_size_bytes=4000000000,
            use_calibration=True,
            max_batch_size=1)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=args.weights,
            conversion_params=conversion_params
        )
        # TODO: representative_data_gen
        converter.convert(calibration_input_fn=representative_data_gen)
    elif args.quantize_model == 'float16':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP16,
            max_workspace_size_bytes=4000000000,
            use_calibration=True,
            max_batch_size=1)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=args.weights,
            conversion_params=conversion_params
        )
        converter.convert()
    elif args.quantize_model == 'float32':
        conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            precision_mode=trt.TrtPrecisionMode.FP32,
            max_workspace_size_bytes=4000000000,
            use_calibration=True,
            max_batch_size=1)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=args.weights,
            conversion_params=conversion_params
        )
        converter.convert()

    converter.save(output_saved_model_dir=args.output_path)
    print('Done Converting to TF-TRT')

    # 加载 trt 模型
    saved_model_loaded = tf.saved_model.load(args.output_path)
    # 获取推理函数
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    trt_graph = graph_func.graph.as_gtaph_def()
    for n in trt_graph.node:
        print(n.op)
        if n.op == "TRTEngineOp":
            print("Node: %s, %s" % (n.op, n.name.replace("/", "_")))
        else:
            print("Exclude Node: %s, %s" % (n.op, n.name.replace("/", "_")))

    trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
    print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
    all_nodes = len([1 for n in trt_graph.node])
    print("numb. of all_nodes in TensorRT graph:", all_nodes)


if __name__=='__main__':
