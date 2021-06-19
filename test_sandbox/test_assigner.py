import os,sys,json
from pathlib import Path
import tensorflow as tf
PATH =Path(os.getcwd()).parent
sys.path.append(str(PATH.absolute()) + "/")
PATH =Path(os.getcwd())
sys.path.append(str(PATH.absolute()) + "/")
from mmdet.core import bbox
import tensorflow as tf
import numpy as np

config = dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
    )
self = bbox.build_assigner(config)
bboxes = tf.convert_to_tensor([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [5, 5, 15, 15],
    [32, 32, 38, 42],
],dtype=tf.float32)
gt_bboxes = tf.convert_to_tensor([
    [0, 0, 10, 9],
    [0, 0, 0, 0],
], dtype=tf.float32)
gt_labels =tf.convert_to_tensor([2, 3])
assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
expected_gt_inds = tf.convert_to_tensor([1, 0, 2, 0])
assert tf.cast(tf.math.reduce_mean(tf.math.abs(assign_result.gt_inds- expected_gt_inds )), tf.float32) ==0
print("out f")
print(assign_result)
config = dict(
    type='MaxIoUAssigner',  
    pos_iou_thr=0.5,
    neg_iou_thr=0.5,
    ignore_iof_thr=0.5,
    ignore_wrt_candidates=False,
)
self = bbox.build_assigner(config)
assign_result = self.assign(bboxes, gt_bboxes, gt_labels=gt_labels)
expected_gt_inds = tf.convert_to_tensor([1, 0, 2, -1])
assert tf.cast(tf.math.reduce_mean(tf.math.abs(assign_result.gt_inds- expected_gt_inds )), tf.float32) == 0


config = dict(
    type='MaxIoUAssigner',  
    pos_iou_thr=0.5,
    neg_iou_thr=0.5,
)
self = bbox.build_assigner(config)
bboxes = tf.convert_to_tensor([
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [5, 5, 15, 15],
        [32, 32, 38, 42],
    ], dtype=tf.float32)
gt_bboxes = tf.ones(shape=(0,4), dtype=tf.float32)

assign_result = self.assign(bboxes, gt_bboxes)

expected_gt_inds = tf.convert_to_tensor([0, 0, 0, 0])
assert tf.cast(tf.math.reduce_mean(tf.math.abs(assign_result.gt_inds- expected_gt_inds )), tf.float32) == 0
print('pass all')