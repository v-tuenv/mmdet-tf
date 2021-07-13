import os,sys,json
from pathlib import Path
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")

from mmdet import core
import tensorflow as tf
tf.config.run_functions_eagerly(False)
config = dict(

    type='DeltaXYWHBBoxCoder',

)
coder = core.build_bbox_coder(config)

rois = tf.convert_to_tensor([[0., 0., 1., 1.], [0., 0., 1., 1.], [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
deltas = tf.convert_to_tensor([[0., 0., 0., 0.], [1., 1., 1., 1.],
                        [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
expected_decode_bboxes = tf.convert_to_tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                                        [0.1409, 0.1409, 2.8591, 2.8591],
                                        [0.0000, 0.3161, 4.1945, 0.6839],
                                        [5.0000, 5.0000, 5.0000, 5.0000]])

out = coder.decode(rois, deltas, max_shape=(32, 32))

assert tf.math.reduce_mean(tf.math.abs(out - expected_decode_bboxes)) < 5e-5


batch_rois =tf.tile(tf.expand_dims(rois,0) ,(2, 1, 1))
batch_deltas =tf.tile( tf.expand_dims(deltas,0),(2, 1, 1))
batch_out = coder.decode(batch_rois, batch_deltas, max_shape=(32, 32))[0]

assert tf.math.reduce_mean(tf.math.abs(out - batch_out)) < 5e-4

config = dict(
    type='DeltaXYWHBBoxCoder',
    add_ctr_clamp=True,
    ctr_clamp=2
)
coder = core.build_bbox_coder(config)
rois = tf.convert_to_tensor([[0., 0., 6., 6.], [0., 0., 1., 1.], [0., 0., 1., 1.],
                         [5., 5., 5., 5.]])
deltas = tf.convert_to_tensor([[1., 1., 2., 2.], [1., 1., 1., 1.],
                        [0., 0., 2., -1.], [0.7, -1.9, -0.5, 0.3]])
expected_decode_bboxes = tf.convert_to_tensor([[0.0000, 0.0000, 27.1672, 27.1672],
                                        [0.1409, 0.1409, 2.8591, 2.8591],
                                        [0.0000, 0.3161, 4.1945, 0.6839],
                                        [5.0000, 5.0000, 5.0000, 5.0000]])

out = coder.decode(rois, deltas, max_shape=(32, 32))
assert tf.math.reduce_mean(tf.math.abs(out - expected_decode_bboxes)) < 5e-4
print('pass all')