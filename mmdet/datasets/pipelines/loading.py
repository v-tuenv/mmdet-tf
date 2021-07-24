from .builder import PIPELINE
import tensorflow as tf
from mmdet.datasets.tfrecords import tf_example_decoder
@PIPELINE.register_module()
class LoadRecord():
    def __init__(self, require_fields =None, decode_cfg=None):
        self.name = 'load_record_from_tfrecord'
        if require_fields is None:
            require_fields = ['image','boxes','classes']
        self.require_fields=require_fields
        if decode_cfg is None:
            decode_cfg = {'include_mask':False}
        self.example_decoder = tf_example_decoder.TfExampleDecoder(
                **decode_cfg
            )
    def __call__(self, value):
        return self.dataset_parser(value)
    @tf.autograph.experimental.do_not_convert
    def dataset_parser(self,value):
        """Parse data to a fixed dimension input image and learning targets.
        Args:
        value: a single serialized tf.Example string.
        example_decoder: TF example decoder.
        anchor_labeler: anchor box labeler.
        params: a dict of extra parameters.
        Returns:
        image: Image tensor that is preprocessed to have normalized value and
            fixed dimension [image_height, image_width, 3]
        cls_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors]. The height_l and width_l
            represent the dimension of class logits at l-th level.
        box_targets_dict: ordered dictionary with keys
            [min_level, min_level+1, ..., max_level]. The values are tensor with
            shape [height_l, width_l, num_anchors * 4]. The height_l and
            width_l represent the dimension of bounding box regression output at
            l-th level.
        num_positives: Number of positive anchors in the image.
        source_id: Source image id. Default value -1 if the source id is empty
            in the groundtruth annotation.
        image_scale: Scale of the processed image to the original image.
        boxes: Groundtruth bounding box annotations. The box is represented in
            [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed
            dimension [self._max_instances_per_image, 4].
        is_crowds: Groundtruth annotations to indicate if an annotation
            represents a group of instances by value {0, 1}. The tensor is
            padded with 0 to the fixed dimension [self._max_instances_per_image].
        areas: Groundtruth areas annotations. The tensor is padded with -1
            to the fixed dimension [self._max_instances_per_image].
        classes: Groundtruth classes annotations. The tensor is padded with -1
            to the fixed dimension [self._max_instances_per_image].
        """
        with tf.name_scope('parser'):
            data = self.example_decoder.decode(value)
            source_id = data['source_id']
            image = data['image']
            boxes = data['groundtruth_boxes']
            classes = data['groundtruth_classes']
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
            areas = data['groundtruth_area']
            is_crowds = data['groundtruth_is_crowd']
            image_masks = data.get('groundtruth_instance_masks', [])
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
        return  {
            'boxes':boxes,
            'image':image,
            'classes':classes
        }

@PIPELINE.register_module()
class LoadFromFile:
    def __init__(self,):
        self.name='load_raw_image'
    def __call__(self, value):
        '''value : str:contain path to image
        '''
        image_file_name = value['image_file_name']
        encoded_jpg_io = tf.io.read_file(image_file_name)
        encoded_jpg_io = tf.io.decode_image(encoded_jpg_io, channels=3)
        return {'image':encoded_jpg_io}
