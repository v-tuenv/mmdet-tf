import os,sys,json
from pathlib import Path
import tensorflow as tf
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")
from mmdet import core
config = dict(type='BboxOverlaps2D', scale=1.)
iou_cal = core.build_iou_calculator(config)
print(iou_cal)
bboxes1 = tf.convert_to_tensor(
    [
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [32, 32, 38, 42],
    ], dtype=tf.float32
)
bboxes2 = tf.convert_to_tensor(
    [
        [0, 0, 10, 20],
        [0, 10, 10, 19],
        [10, 10, 20, 20],
    ], dtype=tf.float32
)
overlaps = iou_cal(bboxes1, bboxes2)
assert overlaps.shape == (3, 3)
overlaps = iou_cal(bboxes1, bboxes2, is_aligned=True)
assert overlaps.shape == (3, )
empty = tf.reshape(tf.convert_to_tensor(()), (0, 4))
nonempty = tf.convert_to_tensor([[0,0,10,9]], dtype=tf.float32)
assert tuple(iou_cal(empty, nonempty).shape) == (0, 1)
assert tuple(iou_cal(nonempty, empty).shape) == (1, 0)
assert tuple(iou_cal(empty, empty).shape) == (0, 0)
print("pass all")
