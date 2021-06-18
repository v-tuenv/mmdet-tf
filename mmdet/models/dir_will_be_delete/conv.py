from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import diag, pad
from tensorflow.python.ops.gen_math_ops import not_equal

from .builder import CONV_LAYERS
from .mix_layers import SequentialLayer

CONV_LAYERS.register_module('Conv1d', module=layers.Conv1D)
CONV_LAYERS.register_module('Conv2d', module=layers.Conv2D)
CONV_LAYERS.register_module('Conv3d', module=layers.Conv3D)
CONV_LAYERS.register_module('Conv', module=layers.Conv2D)

def merger(pre, affter, cfg):
    prex = cfg.pop(pre,None)
    if prex:
        cfg[affter] = prex

def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.
    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.
    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    padding = cfg_.pop("padding",None)
    if padding is None:
        padding = kwargs.pop("padding",None)
    merger("bias",'use_bias',cfg_)
    merger("bias","use_bias",kwargs)
    # if bias:
    #     cfg_['use_bias'] = bias
    #     tf.print("pls refactor code bias to use_bias")
    merger('dilation','dilation_rate',cfg_)
    merger('dilation','dilation_rate',kwargs)
    
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)
    if padding is not None:
        padding_layer = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))
        return SequentialLayer([padding_layer, layer])
    return layer
