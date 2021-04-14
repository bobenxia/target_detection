from tensorflow.keras.layers import ZeroPadding2D, Add, Concatenate
from layers import compose, darknet_CBM


def csp_resblock_body(x, num_filters, num_blocks, all_narrow=True):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # 填充x的边界为0，由(?, 416, 416, 32)转换为(?, 417, 417, 32)。
    # 因为下一步卷积操作的步长为2，所以图的边长需要是奇数。
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    # 第一个CBM对高和宽进行压缩
    x = darknet_CBM(num_filters, (3, 3), strides=(2, 2))(x)  # darknet中只有卷积层，通过调节卷积步长控制输出特征图的尺寸

    # 残差
    res_connection = darknet_CBM(num_filters//2 if all_narrow else num_filters, (1, 1))(x)
    # 主干
    x = darknet_CBM(num_filters//2 if all_narrow else num_filters, (1, 1))(x)
    for i in range(num_blocks):
        x_blocks = compose(
            darknet_CBM(num_filters//2, (1,1)),
            darknet_CBM(num_filters//2 if all_narrow else num_filters, (3, 3)))(x)
        x = Add()([x, x_blocks])

    x = darknet_CBM(num_filters//2 if all_narrow else num_filters, (1, 1))(x)
    x = Concatenate()([x, res_connection])  # 主干、残差汇合
    x = darknet_CBM(num_filters, (1, 1))(x)

    return x


def csp_darknet53_body(x):
    """CSPDarknet53 body having 52 Convolution2D layers"""
    x = darknet_CBM(32, (3, 3))(x)
    x = csp_resblock_body(x, 64, 1, False)
    x = csp_resblock_body(x, 128, 2)
    x = csp_resblock_body(x, 256, 8)
    x = csp_resblock_body(x, 512, 8)
    x = csp_resblock_body(x, 1024, 4)
    return x

