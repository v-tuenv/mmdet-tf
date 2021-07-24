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

from absl import logging
import numpy as np
import PIL.Image
import six
import tensorflow as tf
path = os.path.abspath(__file__)
root = os.path.sep.join(path.split(os.path.sep)[:-2])
import sys
sys.path.append(root)
print(path, root)
from mmdet.datasets.tfrecords.create_simple_tfrecord import create_from_generator



class CreatePascalTFRecordTest(tf.test.TestCase):

  def test_dict_to_tf_example(self):
    label_map_dict = {
        'person':1,
        'not_person':0
    }
    def generator():
        for i in range(10):
            image_file_name = f'2012_12_{i}.jpg'
            image_data = np.random.rand(256, 256, 3)
            save_path = os.path.join(self.get_temp_dir(), image_file_name)
            image = PIL.Image.fromarray(image_data, 'RGB')
            image.save(save_path)
            root_save = self.get_temp_dir()
            data = {
                'image_file_name':save_path,
                'object':[
                    {'bndbox':{'xmin':0,'ymin':12+i,'xmax':200+i,'ymax':232+i},'name':'person'}
                ]
            }
            yield data
    json_anotations = create_from_generator(generator,label_map_dict,self.get_temp_dir(),shard_file=2)
    print(json_anotations)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()