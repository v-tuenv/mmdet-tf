import tensorflow as tf

from .base_detector import BaseDetector

from ..builder import DETECTORS, build_backbone, build_roi_extractor, build_neck, build_head
import warnings
@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.
    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 ):
        super(SingleStageDetector, self).__init__()
        self.init_cfg=init_cfg
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone['pretrained'] = pretrained

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

  
    @tf.function(experimental_relax_shapes=True)
    def extract_feat(self, img,training=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img,training=training)
        if self.with_neck:
            x = self.neck(x,training=training)
        return x
    @tf.function(experimental_relax_shapes=True)
    def call(self, img,training=False):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img,training=training)
        outs = self.bbox_head(x,training=training)
        return outs
   
    def call_funtion(self, img):
        """Used for computing network flops.
        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.backbone.call_funtion(img)
        if self.with_neck:
            x = self.neck.call_funtion(x)
        outs = self.bbox_head.call_funtion(x)
        return outs
    @tf.function(experimental_relax_shapes=True)
    def forward_train(self,
                      img,
                      gt_bboxes,
                      gt_labels,
                      batch_size):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        print("trac forward train")
        super(SingleStageDetector, self).forward_train(img)
        x = self.extract_feat(img,training=True)
        losses = self.bbox_head.forward_train(x, gt_bboxes,
                                              gt_labels, batch_size=batch_size)
        return losses
    def simple_test(self, imgs):
        feat = self.extract_feat(imgs)
        results_list=self.bbox_head.simple_test(feat)
        return results_list
        
    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.
        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)

        pass

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.
        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        pass
    

