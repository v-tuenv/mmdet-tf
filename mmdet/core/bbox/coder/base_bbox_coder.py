from abc import ABCMeta, abstractmethod

import tensorflow as tf
class BaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    @tf.function(experimental_relax_shapes=True)
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    @tf.function(experimental_relax_shapes=True)
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""