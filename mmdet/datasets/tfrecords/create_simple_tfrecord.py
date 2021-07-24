import enum
import os

from absl import logging
import numpy as np
import PIL.Image
import six
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import imag

from . import create_pascal_tfrecord
import hashlib
import io
import json
import os

from absl import app
from absl import flags
from absl import logging
import tqdm
from lxml import etree
import PIL.Image
import tensorflow as tf

# from mmdet.datasets.tf dataset import tfrecord_util
from mmdet.datasets.tfrecords import tf_record_utils as tfrecord_util
from mmdet.datasets.tfrecords import label_map_util as label_map_util

def dict_to_tf_example(data,
                       label_map_dict,
                       unique_id,
                       ignore_difficult_instances=False,
                       ann_json_dict=None):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by running
      tfrecord_util.recursive_parse_xml_to_dict)
    images_dir: Path to the directory holding raw images.
    label_map_dict: A map from string label names to integers ids.
    unique_id: UniqueId object to get the unique {image/ann}_id for the image
      and the annotations.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    ann_json_dict: annotation json dictionary.
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = data['image_file_name']
  with tf.io.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  format =image.format.lower()
  key = hashlib.sha256(encoded_jpg).hexdigest()
  image_id = unique_id.get_image_id()
  size_ = data.pop('size',None)
  if size_ is None:
    size_ ={'width':image.size[0],'height':image.size[1]}
  width = int(size_['width'])
  height = int(size_['height'])
  if ann_json_dict:
    image = {
        'file_name': data['image_file_name'],
        'height': height,
        'width': width,
        'id': image_id,
    }
    ann_json_dict['images'].append(image)

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  area = []
  classes = []
  classes_text = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj.pop('difficult',0)))
      if ignore_difficult_instances and difficult:
        continue
      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      area.append((xmax[-1] - xmin[-1]) * (ymax[-1] - ymin[-1]))
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      if ann_json_dict:
        abs_xmin = int(obj['bndbox']['xmin'])
        abs_ymin = int(obj['bndbox']['ymin'])
        abs_xmax = int(obj['bndbox']['xmax'])
        abs_ymax = int(obj['bndbox']['ymax'])
        abs_width = abs_xmax - abs_xmin
        abs_height = abs_ymax - abs_ymin
        ann = {
            'area': abs_width * abs_height,
            'iscrowd': 0,
            'image_id': image_id,
            'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
            'category_id': label_map_dict[obj['name']],
            'id': unique_id.get_ann_id(),
            'ignore': 0,
            'segmentation': [],
        }
        ann_json_dict['annotations'].append(ann)

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height':
                  tfrecord_util.int64_feature(height),
              'image/width':
                  tfrecord_util.int64_feature(width),
              'image/filename':
                  tfrecord_util.bytes_feature(data['image_file_name'].encode('utf8')),
              'image/source_id':
                  tfrecord_util.bytes_feature(str(image_id).encode('utf8')),
              'image/key/sha256':
                  tfrecord_util.bytes_feature(key.encode('utf8')),
              'image/encoded':
                  tfrecord_util.bytes_feature(encoded_jpg),
              'image/format':
                  tfrecord_util.bytes_feature(format.encode('utf8')),
              'image/object/bbox/xmin':
                  tfrecord_util.float_list_feature(xmin),
              'image/object/bbox/xmax':
                  tfrecord_util.float_list_feature(xmax),
              'image/object/bbox/ymin':
                  tfrecord_util.float_list_feature(ymin),
              'image/object/bbox/ymax':
                  tfrecord_util.float_list_feature(ymax),
              'image/object/area':
                  tfrecord_util.float_list_feature(area),
              'image/object/class/text':
                  tfrecord_util.bytes_list_feature(classes_text),
              'image/object/class/label':
                  tfrecord_util.int64_list_feature(classes),
              'image/object/difficult':
                  tfrecord_util.int64_list_feature(difficult_obj),
          }))
  return example


def create_from_generator(generator,label_map_dict, root_save, shard_file=4):
    '''
        generator : yield data : dict 
        +data_format :
            - image_file_name : file_load_images
            - size : optional : size_images
            - object: list [
                {
                    'bndbox':{
                        'xmin':int,
                        'xmax': int,
                        'ymin':int,
                        'ymax':int
                    },
                    'name':(str:class_name),
                    'difficult':optinal - [1,0] || 1-ignore: 0-not ignore
                }
            ]
    '''
    assert isinstance(label_map_dict, dict)
    annotations = {
        'images': [],
        'type': 'object_detection',
        'annotations': [],
        'categories': []
    }
    unique_id = create_pascal_tfrecord.UniqueId()
    for class_name, class_id in label_map_dict.items():
        cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
        annotations['categories'].append(cls)
    writers = [
        tf.io.TFRecordWriter(os.path.join(root_save,'%05d-of-%05d.tfrecord' %
                            (i, shard_file)))
        for i in range(shard_file)
    ]
    for idx,data in tqdm.tqdm(enumerate(generator())):
        example = dict_to_tf_example(
            data,
            label_map_dict,
            unique_id=unique_id,
            ann_json_dict=annotations
        )
        writers[idx %shard_file].write(example.SerializeToString())
    for writer in writers:
        writer.close()
    
    json_file_path =os.path.join(root_save,'annotations.json')
    with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(annotations, f) 

    
    return annotations

        

