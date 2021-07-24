# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for create_pascal_tfrecord.py."""


import os
import PIL.Image as Image
from absl import logging
from matplotlib.pyplot import imshow
import numpy as np
import PIL.Image
import six
import tensorflow as tf
from tensorflow._api.v2 import data
path = os.path.abspath(__file__)
root = os.path.sep.join(path.split(os.path.sep)[:-2])
import sys
sys.path.append(root)
print(path, root)
from mmdet.datasets.tfrecords.create_simple_tfrecord import create_from_generator
from mmdet.datasets.pipelines import Compose,build_pipeline, transforms
from mmdet.datasets.visualizer import vis_utils
from mmdet.core_tf.common import standart_fields
from mmdet import core,models
from mmdet.utils.util_mixins import Config
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet50V1',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHeadV2',
        num_classes=20,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            # target_means=[0.0, 0.0, 0.0, 0.0],
            # target_stds=[1., 1., 1., 1.],
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=Config(dict(
        assigner=dict(
            type='ArgMaxMatcher',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            iou_calculator=dict(type='IouSimilarity')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)),
    test_cfg=Config(dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)))
# retina = models.build_detector(model)
class PipeLineTest(tf.test.TestCase):
  
  def test_simple_pipeline(self):
    w,h=(256,256)
    tf.config.run_functions_eagerly(False)
    retina = models.build_detector(model)
    print("first call build weights")
    images = tf.random.normal(shape=(2,h,w,3))
    outs=retina(images, training=True)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    datasets = tf.data.Dataset.list_files("./dataset/test_tf/*.tfrecord").shard(2,0)
    def _prefetch_dataset(filename):
        dataset = tf.data.TFRecordDataset(filename).prefetch(1)
        return dataset

    datasets = datasets.interleave(
            _prefetch_dataset, num_parallel_calls=AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    datasets = datasets.with_options(options)

    transform = [
        dict(type='LoadRecord'),
        dict(type='Resize',target_size=(h,w),scale_min=0.8,
                                        scale_max=2.),
        dict(type='PadInstance',num=12)
    ]
    compose_pipeline = Compose(transform)
    map_fn = lambda value:compose_pipeline(value)
    datasets = datasets.map(
    map_fn, num_parallel_calls=AUTOTUNE)
    map_head = retina.bbox_head.get_function_prepare_offline(outs)
    for i in datasets.take(1):
        print(i)
    datasets  = datasets.map(
        map_head, num_parallel_calls=AUTOTUNE
    )
    datasets = datasets.batch(4)
    retina.compile(optimizer='adam')
    # for i in datasets.take(1):
    #     for k in i.keys():
    #         print(k,i[k].shape)
    #     _ = retina.train_step(i)
    retina.fit(datasets,epochs=1, steps_per_epoch=3)
# tf.config.run_functions_eagerly(True)
# retina = models.build_detector(model)
# print("first call build weights")
# images = tf.random.normal(shape=(2,512,512,3))
# outs=retina(images, training=True)
# AUTOTUNE = tf.data.experimental.AUTOTUNE
# datasets = tf.data.Dataset.list_files("./dataset/test_tf/*.tfrecord").shard(2,0)
# def _prefetch_dataset(filename):
#     dataset = tf.data.TFRecordDataset(filename).prefetch(1)
#     return dataset

# datasets = datasets.interleave(
#         _prefetch_dataset, num_parallel_calls=AUTOTUNE)

# options = tf.data.Options()
# options.experimental_deterministic = False
# options.experimental_optimization.map_parallelization = True
# options.experimental_optimization.parallel_batch = True
# datasets = datasets.with_options(options)

# transform = [
#     dict(type='LoadRecord'),
#     dict(type='Resize',target_size=(256,256),scale_min=0.8,
#                                     scale_max=2.),
#     dict(type='PadInstance',num=12)
# ]
# compose_pipeline = Compose(transform)
# map_fn = lambda value:compose_pipeline(value)
# datasets = datasets.map(
# map_fn, num_parallel_calls=AUTOTUNE)
# map_head = retina.bbox_head.get_function_prepare_offline(outs)
# for i in datasets.take(1):
#     print(i)
# datasets  = datasets.map(
#     map_head, num_parallel_calls=AUTOTUNE
# )
# datasets = datasets.batch(4)
# retina.compile(optimizer='adam')
# for i in datasets.take(1):
#     for k in i.keys():
#         print(k,i[k].shape)
#     _ = retina.train_step(i)
if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()