
import math
import os
import re
from absl import logging

import numpy as np

import tensorflow as tf
from tensorflow_addons.callbacks import AverageModelCheckpoint
import tensorflow_hub as hub


from . import coco as coco_metric
import utils
# from keras import anchors
# from keras import efficientdet_keras
# from keras import label_util
# from keras import postprocess
# from keras import util_keras
# from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper


class COCOCallback(tf.keras.callbacks.Callback):
  """A utility for COCO eval callback."""

  def __init__(self, test_dataset, update_freq=None):
    super().__init__()
    self.test_dataset = test_dataset
    self.update_freq = update_freq

  def set_model(self, model: tf.keras.Model):
    self.model = model
    config = model.config
    self.config = config
    label_map = config['label_map']
    log_dir = os.path.join(config['model_dir'], 'coco_cb')
    self.file_writer = tf.summary.create_file_writer(log_dir)
    self.evaluator = coco_metric.EvaluationMetric(
        filename=config['val_json_file'], label_map=label_map)

  @tf.function
  def _get_detections(self, images, labels):
    cls_outputs, box_outputs = self.model(images, training=False)
    detections = postprocess.generate_detections(self.config,
                                                 cls_outputs,
                                                 box_outputs,
                                                 labels['image_scales'],
                                                 labels['source_ids'])
    tf.numpy_function(self.evaluator.update_state,
                      [labels['groundtruth_data'],
                       postprocess.transform_detections(detections)], [])

  def on_epoch_end(self, epoch, logs=None):
    epoch += 1
    if self.update_freq and epoch % self.update_freq == 0:
      self.evaluator.reset_states()
      strategy = tf.distribute.get_strategy()
      count = self.config.eval_samples // self.config.batch_size
      dataset = self.test_dataset.take(count)
      dataset = strategy.experimental_distribute_dataset(dataset)
      for (images, labels) in dataset:
        strategy.run(self._get_detections, (images, labels))
      metrics = self.evaluator.result()
      eval_results = {}
      with self.file_writer.as_default(), tf.summary.record_if(True):
        for i, name in enumerate(self.evaluator.metric_names):
          tf.summary.scalar(name, metrics[i], step=epoch)
          eval_results[name] = metrics[i]
      return eval_results