from yolo import YOLO


def frozen_graph(pb_model_path):
    """
    冻结模型，将训练好的 .h5 模型文件转成 .pb文件
    :param pb_model_path: pb 模型文件保存路径
    :return:
    """
    # 加载模型
    yolo = YOLO()
    model = yolo.inference_model

    model.summary()
    model.save(pb_model_path)


if __name__ == '__main__':
    pb_model_path = '../../weights'
    frozen_graph(pb_model_path)

