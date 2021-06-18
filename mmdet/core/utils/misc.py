from functools import partial

import numpy as np
import tensorflow as tf
from six.moves import map, zip


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of size
    count)"""
    if tf.rank(data) == 1:
        ret = tf.fill((count,),fill)
        ret = tf.cast(ret, dtype=data.dtype)
        ret  = tf.tensor_scatter_nd_update(ret, tf.expand_dims(inds, axis=-1), data)
       
    else:
        new_size = (count, ) + data.shape[1:]
        ret = tf.fill(new_size, fill)
        ret = tf.cast(ret, dtype=data.dtype)
        ret  = tf.tensor_scatter_nd_update(ret, tf.expand_dims(inds, axis=-1), data)
    return ret

    