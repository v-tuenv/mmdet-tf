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
class PipeLineTest(tf.test.TestCase):
  
  def test_simple_pipeline(self):
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
        dict(type='Resize',target_size=(512,640),scale_min=0.8,
                                        scale_max=2.)
    ]
    compose_pipeline = Compose(transform)
    map_fn = lambda value:compose_pipeline(value)
    datasets = datasets.map(
    map_fn, num_parallel_calls=AUTOTUNE)
    a=0
    for i in datasets.take(6):
        image =i[standart_fields.InputDataFields.image]
        print(image.shape)
        h,w = image.shape[0],image.shape[1]
        boxes = i[standart_fields.InputDataFields.groundtruth_boxes].numpy() * np.array([[1./h,1./w,1./h,1./w]]).reshape(-1,4)
        image=image.numpy()
        import cv2
        cv2.imwrite(f"./test_case/resources/{a}_img.png",image)
        vis_utils.draw_bounding_boxes_on_image_array(image, boxes)
        cv2.imwrite(f"./test_case/resources/{a}__img.png",image)
        a =a+1

if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()