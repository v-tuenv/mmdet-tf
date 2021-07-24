from tensorflow.python.data.util import nest
from mmdet.core_tf.anchors import anchor_generator
from operator import gt
from mmdet.core import bbox
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as nn
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.math_ops import reduce_max

from mmdet.core_tf import (build_bbox_coder, build_iou_calculator, build_anchor_generator,
                            build_matcher, build_sampler, build_target_assigner)

# from mmdet.core import (build_bbox_coder, build_anchor_generator,anchor_inside_flags,images_to_levels,
#                          build_sampler, build_assigner, unmap, multi_apply)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead, BaseDenseHeadV2, nested_loss
from mmdet.core_tf.common import box_list, standart_fields 
@HEADS.register_module()
class AnchorHead(BaseDenseHead):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605
    def __init__(
                self,
                num_classes,
                in_channels,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8, 16, 32],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                train_cfg=None,
                test_cfg=None,
                init_cfg=dict(type='Normal', layers='Conv2d', std=0.01),
                compute_target_online=False):
        super(AnchorHead,self).__init__(init_cfg)
        self.in_channels=in_channels
        self.num_classes=num_classes
        self.feat_channels=feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        self.compute_target_online=compute_target_online
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder_cfg = bbox_coder

        self.bbox_coder=build_bbox_coder(bbox_coder)
        self.loss_cls_cfg = loss_cls

        # self.set_attr_serializer('loss_cls_cfg',loss_cls) # this is loss_funtion
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cfg=loss_bbox
        # self.set_attr_serializer('loss_bbox_cfg',loss_bbox)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg  = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner_cfg = self.train_cfg.assigner
            # self.set_attr_serializer('assigner_cfg',self.train_cfg.assigner)
            self.assigner = build_matcher(self.train_cfg.assigner)

            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler_cfg = sampler_cfg
            self.sampler = build_sampler(sampler_cfg)

        self.target_assigner = build_target_assigner( dict(type='TargetAnchorAssigner',
                                                        matcher=self.assigner,
                                                        box_coder_instance=self.bbox_coder,
                                                        sampler=self.sampler,
                                                        negative_class_weight=1.0,
                                                        ignored_value=0.,
                                                        )
        )
        self.fp16_enabled = False
        self.anchor_generator_cfg = anchor_generator

        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors =  self.anchor_generator.num_base_anchors[0]

        # self.set_attr_serializer('num_anchors', self.anchor_generator.num_base_anchors[0] )
        
        self.m_init_layers()

    def m_init_layers(self):
        '''init layers or weights
        '''
        self.conv_cls = nn.Conv2D(self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2D(self.num_anchors * 4, 1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, feats, training=False):
        outs = []
        N = len(feats)
        for i in range(N):
            outs.append(self.forward_single(feats[i],training=training))
        outs=tuple(map(list, zip(*outs)))
        return outs
    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, x,training=False):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x,training=training)
        bbox_pred = self.conv_reg(x,training=training)
        return cls_score, bbox_pred
    def forward_single_function(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred
    def get_anchors(self, featmap_sizes,num_imgs):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        
        if num_imgs is None:
            tf.print("None batch_size compute")
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list
    def call_function(self, feats):
        outs = []
        N = len(feats)
        for i in range(N):
            outs.append(self.forward_single_function(feats[i]))
        outs=tuple(map(list, zip(*outs)))
        return tf.keras.Model(inputs=feats, outputs=outs)
    def mloss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             
             ):
        """Compute losses of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        featmap_sizes = [featmap.shape[-3:-1] for featmap in cls_scores]
        N = cls_scores[0].shape[0]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        for i in range(len(cls_scores)):
            cls_scores[i] = tf.reshape(cls_scores[i],(N,-1))
            bbox_preds[i] = tf.reshape(bbox_preds[i], (N,-1))
        cls_scores = tf.concat(cls_scores, axis=-1) # Batch_size*num_bbox* cls_out_channels
        bbox_preds = tf.concat(bbox_preds, axis=-1) # batch_size*num_bbox,4
        cls_scores = tf.reshape(cls_scores,(-1,self.cls_out_channels))
        bbox_preds = tf.reshape(bbox_preds, (-1,4))

        multi_level_anchors = tf.concat(multi_level_anchors, axis=0) # (num_bbox,4)
        multi_level_anchors = box_list.BoxList(multi_level_anchors)

        for i in range(N):
            gt=tf.where(gt_labels[i] >=0)
            gt=tf.reshape(gt,(-1,))
            gt_labels[i] = tf.gather(gt_labels[i],gt)
            gt_bboxes[i]=tf.gather(gt_bboxes[i],gt)
            gt_labels[i] = tf.reshape(gt_labels[i],(-1,1))
            gt_bboxes[i] =box_list.BoxList(gt_bboxes[i])
        (batch_cls_targets, batch_cls_weights, batch_reg_targets,
                batch_reg_weights, batch_match) = self.batch_assign(multi_level_anchors, gt_bboxes,
                                                                    gt_labels,
                                                                    gt_weights_batch=None) 
        batch_cls_targets = tf.reshape(batch_cls_targets,(-1,))
        batch_cls_weights = tf.reshape(batch_cls_weights,(-1,))
        batch_reg_targets=tf.reshape(batch_reg_targets, (-1,4))
        batch_reg_weights=tf.reshape(batch_reg_weights,(-1,))
        num_positives = tf.reduce_sum(
        tf.cast(tf.greater_equal(batch_match, 0), tf.float32))
        loss_cls = self.loss_cls(
            cls_scores,batch_cls_targets, batch_cls_weights, avg_factor=num_positives)
        loss_bbox = self.loss_bbox(
            bbox_preds,
            batch_reg_targets,
            batch_reg_weights,
            num_positives
            )
        return dict(loss_cls = loss_cls,loss_bbox=loss_bbox)
    
    def get_compute_instance_target(self):
        @tf.autograph.experimental.do_not_convert
        def assign(anchors, gt_boxes, gt_class_targets, unmatched_class_label,
                gt_weights):
            
            pass
        pass

    def batch_assign(self,anchors_batch, gt_box_batch,
                    gt_class_targets_batch,unmatched_class_label=None, gt_weights_batch=None ):
            
        if not isinstance(anchors_batch, list):
            anchors_batch = len(gt_box_batch) * [anchors_batch]
        if not all(
            isinstance(anchors, box_list.BoxList) for anchors in anchors_batch):
            raise ValueError('anchors_batch must be a BoxList or list of BoxLists.')
        if not (len(anchors_batch)
            == len(gt_box_batch)
            == len(gt_class_targets_batch)):
            raise ValueError('batch size incompatible with lengths of anchors_batch, '
                            'gt_box_batch and gt_class_targets_batch.')
        cls_targets_list = []
        cls_weights_list = []
        reg_targets_list = []
        reg_weights_list = []
        match_list = []
        if gt_weights_batch is None:
            gt_weights_batch = [None] * len(gt_class_targets_batch)

        for anchors, gt_boxes, gt_class_targets, gt_weights in zip(
            anchors_batch, gt_box_batch, gt_class_targets_batch, gt_weights_batch):
            (cls_targets, cls_weights,
            reg_targets, reg_weights, match) =self.target_assigner.assign(
                anchors, gt_boxes, gt_class_targets, unmatched_class_label,
                gt_weights)
            cls_targets_list.append(cls_targets)
            cls_weights_list.append(cls_weights)
            reg_targets_list.append(reg_targets)
            reg_weights_list.append(reg_weights)
            match_list.append(match)
        batch_cls_targets = tf.stack(cls_targets_list)
        batch_cls_weights = tf.stack(cls_weights_list)
        batch_reg_targets = tf.stack(reg_targets_list)
        batch_reg_weights = tf.stack(reg_weights_list)
        batch_match = tf.stack(match_list)
        return (batch_cls_targets, batch_cls_weights, batch_reg_targets,
                batch_reg_weights, batch_match)


@HEADS.register_module()
class AnchorHeadV2(BaseDenseHeadV2):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        reg_decoded_bbox (bool): If true, the regression loss would be
            applied directly on decoded bounding boxes, converting both
            the predicted boxes and regression targets to absolute
            coordinates format. Default False. It should be `True` when
            using `IoULoss`, `GIoULoss`, or `DIoULoss` in the bbox head.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605
    def __init__(
                self,
                num_classes,
                in_channels,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8, 16, 32],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                train_cfg=None,
                test_cfg=None,
                init_cfg=dict(type='Normal', layers='Conv2d', std=0.01),
                compute_target_online=False):
        super(AnchorHeadV2,self).__init__(init_cfg)
        self.in_channels=in_channels
        self.num_classes=num_classes
        self.feat_channels=feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        self.compute_target_online=compute_target_online
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder_cfg = bbox_coder

        self.bbox_coder=build_bbox_coder(bbox_coder)
        self.loss_cls_cfg = loss_cls

        # self.set_attr_serializer('loss_cls_cfg',loss_cls) # this is loss_funtion
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cfg=loss_bbox
        # self.set_attr_serializer('loss_bbox_cfg',loss_bbox)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg  = train_cfg
        self.test_cfg = test_cfg

        if self.train_cfg:
            self.assigner_cfg = self.train_cfg.assigner
            # self.set_attr_serializer('assigner_cfg',self.train_cfg.assigner)
            self.assigner = build_matcher(self.train_cfg.assigner)

            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler_cfg = sampler_cfg
            self.sampler = build_sampler(sampler_cfg)

        self.target_assigner = build_target_assigner( dict(type='TargetAnchorAssignerV2',
                                                        matcher=self.assigner,
                                                        box_coder_instance=self.bbox_coder,
                                                        sampler=self.sampler,
                                                        negative_class_weight=1.0,
                                                        ignored_value=0.,
                                                        )
        )

        self.fp16_enabled = False
        self.anchor_generator_cfg = anchor_generator

        self.anchor_generator = build_anchor_generator(anchor_generator)
        self.num_anchors =  self.anchor_generator.num_base_anchors[0]

        # self.set_attr_serializer('num_anchors', self.anchor_generator.num_base_anchors[0] )
        
        self.m_init_layers()
    def prepare_standart_fields(self, data):
        return data
    def get_function_prepare_offline(self, outputs):
        featmap_size = [featmap.shape[-3:-1] for featmap in outputs[1]]
        anchor_generator = self.anchor_generator.grid_anchors(featmap_size)
        self.target_assigner.anchor_generator = anchor_generator
        map_fn = lambda value: self.target_assigner(value)
        return map_fn
    def m_init_layers(self):
        '''init layers or weights
        '''
        self.conv_cls = nn.Conv2D(self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2D(self.num_anchors * 4, 1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, feats, training=False):
        outs = []
        N = len(feats)
        for i in range(N):
            outs.append(self.forward_single(feats[i],training=training))
        outs=tuple(map(list, zip(*outs)))
        return outs
    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, x,training=False):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x,training=training)
        bbox_pred = self.conv_reg(x,training=training)
        return cls_score, bbox_pred
    def forward_single_function(self, x):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_anchors * 4.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        return cls_score, bbox_pred
    def get_anchors(self, featmap_sizes,num_imgs):
        """Get anchors according to feature map sizes.
        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors
        Returns:
            tuple:
                anchor_list (list[Tensor]): Anchors of each image.
                valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        
        if num_imgs is None:
            tf.print("None batch_size compute")
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list
    def call_function(self, feats):
        outs = []
        N = len(feats)
        for i in range(N):
            outs.append(self.forward_single_function(feats[i]))
        outs=tuple(map(list, zip(*outs)))
        return tf.keras.Model(inputs=feats, outputs=outs)
    def mloss(self,
             cls_scores,
             bbox_preds,
             data
             ):
        losses = {
            'loss_cls':[],
            'loss_bbox':[]
        }
        for index,(cls_score, bbox_pred)  in enumerate(zip(cls_scores,bbox_preds)):
            cls_score = tf.reshape(cls_score, (-1, self.cls_out_channels))
            bbox_pred = tf.reshape(bbox_pred, (-1, 4))
            regression_target = data[standart_fields.TargetComputeFields.regression_target+f"_{index}"]
            regression_weight = data[standart_fields.TargetComputeFields.regression_weight+f"_{index}"]
            classification_target = data[standart_fields.TargetComputeFields.classification_target+f"_{index}"]
            classification_weight = data[standart_fields.TargetComputeFields.classification_weight+f"_{index}"]
            num_positives = tf.math.reduce_sum(data[standart_fields.TargetComputeFields.num_positive_fields])
            regression_target = tf.reshape(regression_target,(-1,4))
            regression_weight = tf.reshape(regression_weight,(-1,))
            classification_target = tf.reshape(classification_target, (-1,))
            classification_weight = tf.reshape(classification_weight,(-1,))
            loss_cls = self.loss_cls(
            cls_score,classification_target, classification_weight, avg_factor=num_positives)
            loss_bbox = self.loss_bbox(
                bbox_pred,
                regression_target,
                regression_weight,
                num_positives
                )
            losses['loss_bbox'].append(loss_bbox)
            losses['loss_cls'].append(loss_cls)
        losses['loss_cls'] = sum(losses['loss_cls'])
        losses['loss_bbox'] = sum(losses['loss_bbox'])
        # tf.print("before", losses, nested_loss(losses))
        return losses, nested_loss(losses)

    



