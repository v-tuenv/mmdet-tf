import tensorflow as tf
from collections import OrderedDict
from abc import ABCMeta, abstractmethod

from mmdet.core_tf.common import standart_fields
class BaseDetector(tf.keras.Model,  metaclass=ABCMeta):
    def __init__(self,):
        super().__init__()
    
    @property
    def with_neck(self):
        return hasattr(self,'neck') and self.neck is not None

    
    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))

    @abstractmethod
    def extract_feat(self, imgs):
        """Extract features from images."""
        pass

    def extract_feats(self, imgs):
        """Extract features from multiple images.
        Args:
            imgs (list[torch.Tensor]): A list of images. The images are
                augmented from the same image but in different ways.
        Returns:
            list[torch.Tensor]: Features of different images
        """
        assert isinstance(imgs, list)
        return [self.extract_feat(img) for img in imgs]
    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict,loss=self.forward_train(
                    data
            )
            # tf.print(self.losses,loss_dict,loss)
            loss_dict['loss_additional'] = sum(self.losses)
            loss = loss + loss_dict['loss_additional']
            trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        return loss_dict
        
    @tf.function(experimental_relax_shapes=True)
    def forward_train(self, imgs, **kwargs):
        """
        Args:
            img (list[Tensor]): List of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys, see
                :class:`mmdet.datasets.pipelines.Collect`.
            kwargs (keyword arguments): Specific to concrete implementation.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        # batch_input_shape = tuple(imgs[0].shape[-3:-1])
        # for img_meta in img_metas:
        #     img_meta['batch_input_shape'] = batch_input_shape
        pass

   