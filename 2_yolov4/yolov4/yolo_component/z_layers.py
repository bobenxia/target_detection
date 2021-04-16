from functools import wraps, reduce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Activation
from tensorflow.keras.layers import LeakyReLU, BatchNormalization
from tensorflow.keras.regularizers import l2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.
    使用 Python 的 Lambda 表达式，顺次执行函数列表，且前一个函数的输出是后一个函数的输入。
    compose函数适用于在神经网络中连接两个层。

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def custom_batchnormalization(*args, **kwargs):
    if tf.__version__ >= '2.2':
        from tensorflow.keras.layers.experimental import SyncBatchNormalization
        bn = SyncBatchNormalization
    else:
        bn = BatchNormalization

    return bn(*args, **kwargs)


def mish(x):
    return x * K.tanh(K.softplus(x))


@wraps(Conv2D)
def darknet_Conv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['bias_regularizer'] = l2(5e-4)
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_CBM(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and Mish."""
    no_bias_kwargs = {'use_bias': False}  # 没懂为啥用 no_bias
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_Conv2D(*args, **no_bias_kwargs),
        custom_batchnormalization(),
        Activation(mish))


def darknet_CBL(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""

    no_bias_kwargs = {'use_bias': False}  # 没懂为啥用 no_bias
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_Conv2D(*args, **no_bias_kwargs),
        custom_batchnormalization(),
        LeakyReLU(alpha=0.1)
    )


def spp(x):
    y1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    y2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    y3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)

    y = Concatenate()([y3, y2, y1, x])
    return y


def make_five_darknet_CBL(x, num_filters):
    # 五次卷积
    x = compose(
        darknet_CBL(num_filters, (1, 1)),
        darknet_CBL(num_filters * 2, (3, 3)),
        darknet_CBL(num_filters, (1, 1)),
        darknet_CBL(num_filters * 2, (3, 3)),
        darknet_CBL(num_filters, (1, 1))
    )(x)
    return x


def make_three_darknet_CBL(x, num_filters):
    # 三次卷积
    x = compose(
        darknet_CBL(num_filters, (1, 1)),
        darknet_CBL(num_filters * 2, (3, 3)),
        darknet_CBL(num_filters, (1, 1)))(x)
    return x
