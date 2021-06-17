import os,sys,json
from pathlib import Path
import tensorflow as tf
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")
from mmdet import core

config_anchor_generator = dict(
    type='AnchorGenerator',
    strides=[16,32],
    ratios=[1.,],
    scales=[1.,],
    base_sizes=[9, 18],
)

anchor_generator = core.build_anchor_generator(config_anchor_generator)
all_anchors = anchor_generator.grid_anchors([(2, 2), (1, 1)])

print(anchor_generator)
print(all_anchors)
all_anchors_str =   '''[tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                        [11.5000, -4.5000, 20.5000,  4.5000],
                        [-4.5000, 11.5000,  4.5000, 20.5000],
                        [11.5000, 11.5000, 20.5000, 20.5000]]), \
                        tensor([[-9., -9., 9., 9.]])]
                    '''
print("expected ", all_anchors_str)

values = [tf.convert_to_tensor([[-4.5, -4.5,  4.5,  4.5],
       [11.5, -4.5, 20.5,  4.5],
       [-4.5, 11.5,  4.5, 20.5],
       [11.5, 11.5, 20.5, 20.5]], dtype=tf.float32),
       tf.convert_to_tensor([[-9., -9.,  9.,  9.]], dtype=tf.float32)
       ]

assert tf.math.reduce_mean(tf.math.abs(values[0] -all_anchors[0] )) < 1e-6,  Exception("failed")
assert tf.math.reduce_mean(tf.math.abs(values[1] -all_anchors[1] )) < 1e-6,  Exception("failed")

print("pass all")