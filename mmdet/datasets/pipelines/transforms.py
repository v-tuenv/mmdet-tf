from .builder import PIPELINE

import tensorflow as tf
from mmdet.core_tf.common import box_list, preprocessor
@PIPELINE.register_module()
class Resize:
    def __init__(self, name = 'resize_pipeline', scale_min=1.,
                                        scale_max=1., target_size=None, method='bilinear'):
        self.name=name
        if target_size is None:
            target_size = (640,640)
        self.target_size=target_size
        self.scale = (scale_min, scale_max)
        if method =='bilinear':
            self.method=tf.image.ResizeMethod.BILINEAR
        else:
            raise ValueError(method)
    def set_training_random_scale_factors(self,_image):
        """Set the parameters for multiscale training.
        Notably, if train and eval use different sizes, then target_size should be
        set as eval size to avoid the discrency between train and eval.
        Args:
        scale_min: minimal scale factor.
        scale_max: maximum scale factor.
        target_size: targeted size, usually same as eval. If None, use train size.
        """
        target_size = self.target_size
        scale_min,scale_max = self.scale
        # Select a random scale factor.
        random_scale_factor = tf.random.uniform([], scale_min, scale_max)
        scaled_y = tf.cast(random_scale_factor * target_size[0], tf.int32)
        scaled_x = tf.cast(random_scale_factor * target_size[1], tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        height = tf.cast(tf.shape(_image)[0], tf.float32)
        width = tf.cast(tf.shape(_image)[1], tf.float32)
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - target_size[0], tf.float32)
        offset_x = tf.cast(scaled_width - target_size[1], tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
        return (image_scale,scaled_height,scaled_width,offset_x,offset_y)
    def clip_boxes(self, boxes):
        """Clip boxes to fit in an image."""
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        ymin = tf.clip_by_value(ymin, 0, self.target_size[0] - 1)
        xmin = tf.clip_by_value(xmin, 0, self.target_size[1] - 1)
        ymax = tf.clip_by_value(ymax, 0, self.target_size[0] - 1)
        xmax = tf.clip_by_value(xmax, 0, self.target_size[1] - 1)
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        return boxes
    def __call__(self, value):
        
        (image_scale,_scaled_height
        ,_scaled_width,_crop_offset_x,
        _crop_offset_y)=self.set_training_random_scale_factors(value['image'])
        dtype = value['image'].dtype
        scaled_image = tf.image.resize(
            value['image'], [_scaled_height, _scaled_width], method=self.method)
        scaled_image = scaled_image[_crop_offset_y:_crop_offset_y +
                                    self.target_size[0],
                                    _crop_offset_x:_crop_offset_x +
                                    self.target_size[1], :]
        output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                    self.target_size[0],
                                                    self.target_size[1])
        _image = tf.cast(output_image, dtype)



        boxlist = box_list.BoxList(value['boxes'])
        # boxlist is in range of [0, 1], so here we pass the scale_height/width
        # instead of just scale.
        boxes = preprocessor.box_list_scale(boxlist, _scaled_height,
                                            _scaled_width).get()
        # Adjust box coordinates based on the offset.
        box_offset = tf.stack([
            _crop_offset_y,
            _crop_offset_x,
            _crop_offset_y,
            _crop_offset_x,
        ])
        boxes -= tf.cast(tf.reshape(box_offset, [1, 4]), tf.float32)
        # Clip the boxes.
        boxes = self.clip_boxes(boxes)
        # Filter out ground truth boxes that are illegal.
        indices = tf.where(
            tf.not_equal((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                        0))
        boxes = tf.gather_nd(boxes, indices)
        classes = tf.gather_nd(value['classes'], indices)
        return {'image':_image,'boxes':boxes,'classes':classes}# boxes, classes
