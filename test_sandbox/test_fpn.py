import tensorflow as tf
import numpy as np

import pandas as pd
import json
import sys
sys.path.append("/home/tuenguyen/Desktop/long_pro/mmdet_tf/")
from mmdet.core import *
tf.random.set_seed(12)

from mmdet.models.necks.fpn import FPNTF
from mmdet.models import build_neck
config = dict(
    type='FPNTF',
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=1,
    add_extra_convs='on_input',
    relu_before_extra_convs=True,
    num_outs=5,
    return_funtion=True
)
fpn = build_neck(config)
print(fpn.summary())