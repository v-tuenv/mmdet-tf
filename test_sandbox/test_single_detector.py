import tensorflow as tf
import numpy as np

import pandas as pd
import json
import sys
sys.path.append("/home/tuenguyen/Desktop/long_pro/mmdet_tf/")
from mmdet.core import *
tf.random.set_seed(12)
bboxes=tf.convert_to_tensor([[[ 34.8102   , 122.88     , 141.41643  , 189.952    ],
        [  5.8016996,   6.144    , 255.27478  , 254.976    ],
        [  0.       ,   0.       ,   64.       ,   128.    ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ]],

       [[106.220894 , 102.4      , 158.18507  , 154.112    ],
         [ 62.976    , 105.81333  , 110.08     , 133.12     ],
      
        [122.368    , 106.496    , 157.184    , 139.94667  ],
        [  6.656    , 196.09853  ,  43.008    , 228.25616  ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ],
        [  0.       ,   0.       ,   0.       ,   0.       ]],

       ],dtype=tf.float32)

cate=tf.convert_to_tensor([[2, 1, 3, -1, -1, -1, -1, -1, -1, -1],
       [2, 4, 0, 1, -1, -1, -1, -1, -1, -1],
       ], dtype=tf.int32)

print(bboxes.shape, cate.shape)


from mmdet import core,models
from mmdet.utils.util_mixins import Config
from mmdet import core,models
from mmdet.utils.util_mixins import Config
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNetKeras',
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
        type='RetinaHeadSpaceSTORM',
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
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.]),
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
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='BboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)),
    test_cfg=Config(dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)))
retina = models.build_detector(model)


# retina = models.build_detector(model)

fake_images = tf.random.normal(shape=(2, 512, 512, 3))
tf.config.run_functions_eagerly(True)
a = retina.forward_train(fake_images,
                      bboxes,
                      cate,
                      batch_size=2)
tf.print(a)

tf.print(sum(a['loss_cls']),sum(a['loss_bbox']))

