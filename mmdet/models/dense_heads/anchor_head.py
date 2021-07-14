from mmdet.core import bbox
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as nn
from tensorflow.python.ops.gen_array_ops import size
from tensorflow.python.ops.math_ops import reduce_max

from mmdet.core import (build_bbox_coder, build_anchor_generator,anchor_inside_flags,images_to_levels,
                         build_sampler, build_assigner, unmap, multi_apply)
from ..builder import HEADS, build_loss
from .base_dense_heade import BaseDenseHead, BaseDenseHeadSpaceSTORM

@HEADS.register_module()
class AnchorHeadSpaceSTORM(BaseDenseHeadSpaceSTORM):
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
                init_cfg=dict(type='Normal', layers='Conv2d', std=0.01)):
        super(AnchorHeadSpaceSTORM,self).__init__(init_cfg)
        self.in_channels=in_channels
        self.num_classes=num_classes
        self.feat_channels=feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        # self.set_attr_serializer('in_channels',in_channels)
        # self.set_attr_serializer('num_classes', num_classes)
        # self.set_attr_serializer('feat_channels',feat_channels)
        # self.set_attr_serializer('use_sigmoid_cls',loss_cls.get('use_sigmoid', False))
        # self.set_attr_serializer('sampling', loss_cls['type'] not in [
        #     'FocalLoss', 'GHMC', 'QualityFocalLoss'
        # ])
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
            # self.set_attr_serializer('cls_out_channels', num_classes)
        else:
            self.cls_out_channels = num_classes + 1
            # self.set_attr_serializer('cls_out_channels', num_classes+1)
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        self.bbox_coder_cfg = bbox_coder
        self.bbox_coder=build_bbox_coder(bbox_coder)
        # self.set_attr_serializer('reg_decoded_bbox',reg_decoded_bbox)
        # self.set_attr_serializer('bbox_coder_cfg',bbox_coder)
        # self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls_cfg = loss_cls

        # self.set_attr_serializer('loss_cls_cfg',loss_cls) # this is loss_funtion
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_cfg=loss_bbox
        # self.set_attr_serializer('loss_bbox_cfg',loss_bbox)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg  = train_cfg
        self.test_cfg = test_cfg
        # self.set_attr_serializer('train_cfg',train_cfg)
        # self.set_attr_serializer('test_cfg',test_cfg)
        
        if self.train_cfg:
            self.assigner_cfg = self.train_cfg.assigner
            # self.set_attr_serializer('assigner_cfg',self.train_cfg.assigner)
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler_cfg = sampler_cfg
            # self.set_attr_serializer('sampler_cfg',sampler_cfg)
            self.sampler = build_sampler(sampler_cfg, context=self)
            # todos 
        # self.fp16_enabled=False
        # self.set_attr_serializer('fp16_enabled',False)
        self.fp16_enabled = False
        self.anchor_generator_cfg = anchor_generator
        # self.set_attr_serializer('anchor_generator_cfg',anchor_generator)
        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors =  self.anchor_generator.num_base_anchors[0]
        # self.set_attr_serializer('num_anchors', self.anchor_generator.num_base_anchors[0] )
        
        self.m_init_layers()

    def m_init_layers(self):
        '''init layers or weights
        '''
        self.conv_cls = nn.Conv2D(self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2D(self.num_anchors * 4, 1)
#     def build(self, feats):
#         inputs = [tf.keras.layers.Input(shape=(None,None,i[-1])) for i in feats]
#         self.m_init_layers()
#         outs = []
#         N = len(feats)
#         for i in range(N):
#             outs.append(self.forward_single(inputs[i]))
#         outs=tuple(map(list, zip(*outs)))
#         self.call_fn_wraper = tf.keras.Model(inputs=inputs, outputs=outs)
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
       
    def get_target_one_batch(self, anchors,gt_bboxes,
                            gt_labels,label_channels ):

        assigned_gt_inds, max_overlaps, assigned_labels = self.assigner.assign(
            anchors, gt_bboxes, 
            None if self.sampling else gt_labels)

        pos_inds, neg_inds= self.sampler.sample(assigned_gt_inds, max_overlaps,
                                                assigned_labels, anchors,
                                               gt_bboxes)

        # bbox target
        bbox_weights = tf.reshape(pos_inds,[-1,1])
        if not self.reg_decoded_bbox:
            pos_bbox_targets = tf.concat([tf.convert_to_tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]],tf.float32),
                                        gt_bboxes],axis=0)
            pos_bbox_targets = tf.gather(pos_bbox_targets, assigned_gt_inds+1)
            bbox_targets=self.bbox_coder.encode(
                    anchors,pos_bbox_targets)
        else:
            pos_bbox_targets = tf.concat([tf.convert_to_tensor([[0.,0.,1.,1.],[0.,0.,1.,1.]],tf.float32),
                                        gt_bboxes],axis=0)
            bbox_targets = tf.gather(pos_bbox_targets, assigned_gt_inds+1)
        

        # compute class
        if gt_labels is None:
            tf.print('raise implement rpn seperate')
        labels = assigned_labels
        label_weights = pos_inds + neg_inds
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)
    
    def get_targets(self,
                    anchor_list,
                    gt_bboxes_list,
                    gt_labels_list=None,
                    label_channels=1
                    ):
        """Compute regression and classification targets for anchors in
        multiple images.
        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(anchor_list)
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results=[]
        for i in range(num_imgs):
            
            re_sig = self.get_target_one_batch(
                                            concat_anchor_list[i], gt_bboxes_list[i],
                                            gt_labels_list[i], label_channels=label_channels) 
            results.append(re_sig)
        results = tuple(map(list, zip(*results)))

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list) = results[:6]
        num_total_pos = sum([tf.math.maximum(tf.math.reduce_sum(inds), 1) for inds in pos_inds_list])
        num_total_neg = sum([tf.math.maximum(tf.math.reduce_sum(inds), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        # print(all_label_weights)
        # print(num_level_anchors)
        # num_level_anchors = tf.constant(num_level_anchors)
        labels_list =        images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)

        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        return res 

    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels =tf.reshape(labels, (-1,))
        label_weights = tf.reshape(label_weights,(-1,))
        print(label_weights)
        cls_score =tf.reshape(cls_score,(-1, self.cls_out_channels))
        print(cls_score)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        
        # regression loss
        bbox_targets =tf.reshape(bbox_targets,(-1, 4))
        bbox_weights = tf.reshape(bbox_weights,(-1,))
        bbox_pred =tf.reshape(bbox_pred, (-1,4))# bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors =tf.reshape(anchors,(-1, 4))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
#             bbox_targets = tf.stop_gradient(bbox_targets)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def mloss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,):
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
        # assert len(featmap_sizes) == self.anchor_generator.num_levels
        # print(gt_bboxes)
        # device = cls_scores[0].device
        # print("mloss")
        anchor_list = self.get_anchors(featmap_sizes, N)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            gt_bboxes,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
#         print(cls_reg_targets)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]

     
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))


        all_anchor_list =   images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_clss=[]
        losses_bboxs=[]
        for i in range(len(cls_scores)):
            losses_cls, losses_bbox = self.loss_single(cls_scores[i], bbox_preds[i],
                            all_anchor_list[i],labels_list[i],label_weights_list[i],bbox_targets_list[i],
                            bbox_weights_list[i],num_total_samples=num_total_samples)
            losses_clss.append(losses_cls)
            losses_bboxs.append(losses_bbox)
            # print(losses_cls)
        return dict(loss_cls=losses_clss, loss_bbox=losses_bboxs)
    
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-3:-1] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        mlvl_cls_scores = [cls_scores[i] for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i] for i in range(num_levels)]

        img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list
    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.
        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
        #     mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor =tf.convert_to_tensor(cfg.get('nms_pre', -1), dtype=tf.int32)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score =tf.reshape(cls_score,(batch_size, -1,
                                                     self.cls_out_channels))
            if self.use_sigmoid_cls:
                scores = tf.math.sigmoid(cls_score)
            else:
                scores = tf.math.softmax(cls_score,axis=-1) 
            bbox_pred =tf.reshape(bbox_pred,(batch_size,-1,4)) 
            anchors =tf.broadcast_to(anchors,bbox_pred.shape)
            # Always keep topk op for dynamic input in onnx
        #     from mmdet.core.export import get_k_for_topk
        #     nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
        #     if nms_pre > 0:
        #         # Get maximum scores for foreground classes.
        #         if self.use_sigmoid_cls:
        #             max_scores, _ = scores.max(-1)
        #         else:
        #             # remind that we set FG labels to [0, num_class-1]
        #             # since mmdet v2.0
        #             # BG cat_id: num_class
        #             max_scores, _ = scores[..., :-1].max(-1)

        #         _, topk_inds = max_scores.topk(nms_pre)
        #         batch_inds = torch.arange(batch_size).view(
        #             -1, 1).expand_as(topk_inds)
        #         anchors = anchors[batch_inds, topk_inds, :]
        #         bbox_pred = bbox_pred[batch_inds, topk_inds, :]
        #         scores = scores[batch_inds, topk_inds, :]

        #     bboxes = self.bbox_coder.decode(
        #         anchors, bbox_pred, max_shape=img_shapes)
        #     mlvl_bboxes.append(bboxes)
        #     mlvl_scores.append(scores)

        # batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        # if rescale:
        #     batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
        #         scale_factors).unsqueeze(1)
        # batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        # if torch.onnx.is_in_onnx_export() and with_nms:
        #     from mmdet.core.export import add_dummy_nms_for_onnx
        #     # ignore background class
        #     if not self.use_sigmoid_cls:
        #         num_classes = batch_mlvl_scores.shape[2] - 1
        #         batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
        #     max_output_boxes_per_class = cfg.nms.get(
        #         'max_output_boxes_per_class', 200)
        #     iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        #     score_threshold = cfg.score_thr
        #     nms_pre = cfg.get('deploy_nms_pre', -1)
        #     return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
        #                                   max_output_boxes_per_class,
        #                                   iou_threshold, score_threshold,
        #                                   nms_pre, cfg.max_per_img)
        # if self.use_sigmoid_cls:
        #     # Add a dummy background class to the backend when using sigmoid
        #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        #     # BG cat_id: num_class
        #     padding = batch_mlvl_scores.new_zeros(batch_size,
        #                                           batch_mlvl_scores.shape[1],
        #                                           1)
        #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        # if with_nms:
        #     det_results = []
        #     for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
        #                                           batch_mlvl_scores):
        #         det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
        #                                              cfg.score_thr, cfg.nms,
        #                                              cfg.max_per_img)
        #         det_results.append(tuple([det_bbox, det_label]))
        # else:
        #     det_results = [
        #         tuple(mlvl_bs)
        #         for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
        #     ]
        # return det_results
        tf.print("implement inf")


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

    def __init__(self,
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
                 init_cfg=dict(type='Normal', layers='Conv2d', std=0.01)):
        super(AnchorHead, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        # TODO better way to determine whether sample or not
        self.sampling = loss_cls['type'] not in [
            'FocalLoss', 'GHMC', 'QualityFocalLoss'
        ]
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # use PseudoSampler when sampling is False
            if self.sampling and hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        self.fp16_enabled = False

        self.anchor_generator = build_anchor_generator(anchor_generator)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        self.m_init_layers()

    def m_init_layers(self):
        self.conv_cls = nn.Conv2D(self.num_anchors * self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2D(self.num_anchors * 4, 1)
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
    @tf.function(experimental_relax_shapes=True)
    def call(self, feats, training=False):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        print('trace call')
        return multi_apply(self.forward_single, feats,training=training)
    
    def call_funtion(self, feats, training=False):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple: A tuple of classification scores and bbox prediction.
                - cls_scores (list[Tensor]): Classification scores for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all \
                    scale levels, each is a 4D-tensor, the channels number \
                    is num_anchors * 4.
        """
        print('trace call')
        out = []
        out2=[]
        for v in feats:
            cls_score = self.conv_cls(v)
            bbox_pred = self.conv_reg(v)
            out.append(cls_score)
            out2.append(bbox_pred)
#             return cls_score, bbox_pred
        return out,out2
#         return multi_apply(self.forward_single, feats,training=training)
    
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

    @tf.function(experimental_relax_shapes=True)
    def _get_targets_single(self,
                            anchors,
                            gt_bboxes,
                            gt_labels,
                            label_channels=1,
                           ):
        """Compute regression and classification targets for anchors in a
        single image.
        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        # inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
        #                                    img_meta['img_shape'][:2],
        #                                    self.train_cfg.allowed_border)

        # todos add any() flag
        # if not inside_flags.any():
        #     return (None, ) * 7
        # assign gt and sample anchors
        # inds_inside = tf.reshape(tf.where(inside_flags),(-1,))
        # anchors =tf.gather(flat_anchors, inds_inside)# flat_anchors[inside_flags, :]
       
        # print(anchors)
        # print(gt_bboxes,"trace 204 head_anchor")
        assign_result = self.assigner.assign(
            anchors, gt_bboxes,
            None if self.sampling else gt_labels)
        pos_inds, neg_inds= self.sampler.sample(*assign_result, anchors,
                                              gt_bboxes)

        cate_match_ids = assign_result[0]
        bbox_targets =anchors
        bbox_weights =tf.zeros_like(anchors)

        if not self.reg_decoded_bbox:
            pos_bbox_targets = tf.concat([tf.convert_to_tensor([[1.,1.,1.,1.],[1.,1.,1.,1.]],tf.float32),
                                        gt_bboxes],axis=0)
            pos_bbox_targets = tf.gather(pos_bbox_targets, cate_match_ids+1)
            bbox_targets=self.bbox_coder.encode(
                    anchors,pos_bbox_targets)
            bbox_weights = tf.reshape(pos_inds,[-1,1])
            bbox_weights = tf.tile(tf.reshape(pos_inds,[-1,1]),[1,4])
        else:
            pos_bbox_targets = tf.concat([tf.convert_to_tensor([[1.,1.,2.,2.],[1.,1.,2.,2.]],tf.float32),
                                        gt_bboxes],axis=0)
            bbox_targets = tf.gather(pos_bbox_targets, cate_match_ids+1)
            bbox_weights = tf.reshape(pos_inds,[-1,1])
            
            bbox_weights = tf.tile(tf.reshape(pos_inds,[-1,1]),[1,4])

        if gt_labels is None:
            tf.print('raise implement rpn seperate')
        labels = assign_result[-1]
        label_weights = pos_inds + neg_inds
#         tf.print("labels_weights :\n",tf.math.reduce_sum(pos_inds),tf.math.reduce_sum(neg_inds))
#         label_weights = tf.stop_gradient(label_weights)
#         bbox_weights = tf.stop_gradient(bbox_weights)
        # label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        # print(bbox_targets.shape,bbox_weights.shape)
        # print(label_weights.shape, labels.shape)
        # print('o')
        
        
        # print(pos_inds,pos_inds.shape)
        # print(neg_inds, neg_inds.shape)
        # print('trace size pos_inds')
        # if tf.size(pos_inds) > 0:
        #     if not self.reg_decoded_bbox:
        #         pos_bbox_targets = self.bbox_coder.encode(
        #             sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
        #     else:
        #         pos_bbox_targets = sampling_result.pos_gt_bboxes
        #     pos_inds_e = tf.expand_dims(pos_inds,axis=-1)
        #     bbox_targets=tf.tensor_scatter_nd_update(bbox_targets,\
        #                                             pos_inds_e,\
        #                                             pos_bbox_targets
        #                                             )
        #     bbox_weights = tf.tensor_scatter_nd_add(bbox_weights,\
        #                                             pos_inds_e,\
        #                                             tf.fill((pos_inds.shape[0],4), 1.,)
        #                                             )

        #     if gt_labels is None:
        #         # Only rpn gives gt_labels as None
        #         # Foreground is the first class since v2.5.0
        #         labels = tf.tensor_scatter_nd_update(labels, pos_inds_e, tf.fill(pos_inds.shape, 0))
        #         # labels[pos_inds] = 0
        #     else:
        #         as_gt = tf.gather(gt_labels, sampling_result.pos_assigned_gt_inds)
        #         labels = tf.tensor_scatter_nd_update(labels, pos_inds_e, as_gt)
        #         # labels[pos_inds] = gt_labels[
        #         #     sampling_result.pos_assigned_gt_inds]

        #     if self.train_cfg.pos_weight <= 0:
        #         label_weights = tf.tensor_scatter_nd_update(label_weights, pos_inds_e, tf.fill(pos_inds.shape,1.))
        #         # label_weights[pos_inds] = 1.0
        #     else:
        #         label_weights = tf.tensor_scatter_nd_update(label_weights, pos_inds_e, tf.fill(pos_inds.shape,self.train_cfg.pos_weight))
                
        # if tf.size(neg_inds) > 0:
        #     neg_inds_e = tf.expand_dims(neg_inds, axis=1)
        #     label_weights = tf.tensor_scatter_nd_update(label_weights, neg_inds_e, tf.fill(neg_inds.shape,1.))
            # label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        # if unmap_outputs:
        #     num_total_anchors = flat_anchors.shape[0]
        #     labels = unmap(
        #         labels, num_total_anchors, inds_inside,
        #         fill=self.num_classes)  # fill bg label
        #     label_weights = unmap(label_weights, num_total_anchors,
        #                           inds_inside)
        #     bbox_targets = unmap(bbox_targets, num_total_anchors, inds_inside)
        #     bbox_weights = unmap(bbox_weights, num_total_anchors, inds_inside)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, assign_result)
    @tf.function(experimental_relax_shapes=True)
    def get_targets(self,
                    anchor_list,
                    gt_bboxes_list,
                    gt_labels_list=None,
                    label_channels=1,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.
        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(anchor_list)
        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        concat_anchor_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        results=[]
        for i in range(num_imgs):
            
            re_sig = self._get_targets_single(
                                            concat_anchor_list[i], gt_bboxes_list[i],
                                            gt_labels_list[i], label_channels=label_channels) 
            results.append(re_sig)
        results = tuple(map(list, zip(*results)))
        #     # tuple(map(list, zip(*map_results)))
        # results = multi_apply(
        #     self._get_targets_single,
        #     concat_anchor_list,
        #     gt_bboxes_list,
        #     gt_labels_list,
        #     label_channels=label_channels,
        #     )

        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        # if any([labels is None for labels in all_labels]):
        #     return None
        # sampled anchors of all images
        num_total_pos = sum([tf.math.maximum(tf.math.reduce_sum(inds), 1) for inds in pos_inds_list])
        num_total_neg = sum([tf.math.maximum(tf.math.reduce_sum(inds), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        # print(all_label_weights)
        # print(num_level_anchors)
        # num_level_anchors = tf.constant(num_level_anchors)
        labels_list =        images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        # print([i.shape for i in label_weights_list])
        # print([i.shape for i in labels_list])
        # print([i.shape for i in bbox_targets_list])
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i in range(len(rest_results)):# enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(rest_results[i], num_level_anchors)

        return res + tuple(rest_results)

    @tf.function(experimental_relax_shapes=True)
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.
        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # classification loss
        labels =tf.reshape(labels, (-1,))
        label_weights = tf.reshape(label_weights,(-1,))

        cls_score =tf.reshape(cls_score,(-1, self.cls_out_channels))
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets =tf.reshape(bbox_targets,(-1, 4))
        bbox_weights = tf.reshape(bbox_weights,(-1, 4))
        bbox_pred =tf.reshape(bbox_pred, (-1,4))# bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors =tf.reshape(anchors,(-1, 4))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
#             bbox_targets = tf.stop_gradient(bbox_targets)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    @tf.function(experimental_relax_shapes=True)
    def mloss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,):
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
        # assert len(featmap_sizes) == self.anchor_generator.num_levels
        # print(gt_bboxes)
        # device = cls_scores[0].device
        
        anchor_list = self.get_anchors(featmap_sizes, N)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            gt_bboxes,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
#         print(cls_reg_targets)
        
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        num_level_anchors = [anchors.shape[0] for anchors in anchor_list[0]]
        # concat all level anchors and flagsto a single tensor
        # concat_anchor_list = []
     
        
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))

        # for i in range(len(anchor_list)):
        #     concat_anchor_list.append(tf.concat(anchor_list[i], axis=0))
        print("back to main loss")
        # print(concat_anchor_list)
        all_anchor_list =   images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        # print(all_anchor_list)
        # print(num_level_anchors)
        losses_clss=[]
        losses_bboxs=[]
        for i in range(len(cls_scores)):
            losses_cls, losses_bbox = self.loss_single(cls_scores[i], bbox_preds[i],
                            all_anchor_list[i],labels_list[i],label_weights_list[i],bbox_targets_list[i],
                            bbox_weights_list[i],num_total_samples=num_total_samples)
            losses_clss.append(losses_cls)
            losses_bboxs.append(losses_bbox)

        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_single,
        #     cls_scores,
        #     bbox_preds,
        #     all_anchor_list,
        #     labels_list,
        #     label_weights_list,
        #     bbox_targets_list,
        #     bbox_weights_list,
        #     num_total_samples=num_total_samples,
        #     )
        return dict(loss_cls=losses_clss, loss_bbox=losses_bboxs)
    
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.
        Args:
            cls_scores (list[Tensor]): Box scores for each level in the
                feature pyramid, has shape
                (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each
                level in the feature pyramid, has shape
                (N, num_anchors * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-3:-1] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes)
        mlvl_cls_scores = [cls_scores[i] for i in range(num_levels)]
        mlvl_bbox_preds = [bbox_preds[i] for i in range(num_levels)]

        img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        if with_nms:
            # some heads don't support with_nms argument
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale)
        else:
            result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
                                           mlvl_anchors, img_shapes,
                                           scale_factors, cfg, rescale,
                                           with_nms)
        return result_list
    def _get_bboxes(self,
                    mlvl_cls_scores,
                    mlvl_bbox_preds,
                    mlvl_anchors,
                    img_shapes,
                    scale_factors,
                    cfg,
                    rescale=False,
                    with_nms=True):
        """Transform outputs for a batch item into bbox predictions.
        Args:
            mlvl_cls_scores (list[Tensor]): Each element in the list is
                the scores of bboxes of single level in the feature pyramid,
                has shape (N, num_anchors * num_classes, H, W).
            mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
                bboxes predictions of single level in the feature pyramid,
                has shape (N, num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Each element in the list is
                the anchors of single level in feature pyramid, has shape
                (num_anchors, 4).
            img_shapes (list[tuple[int]]): Each tuple in the list represent
                the shape(height, width, 3) of single image in the batch.
            scale_factors (list[ndarray]): Scale factor of the batch
                image arange as list[(w_scale, h_scale, w_scale, h_scale)].
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
        #     mlvl_anchors)
        batch_size = mlvl_cls_scores[0].shape[0]
        # convert to tensor to keep tracing
        nms_pre_tensor =tf.convert_to_tensor(cfg.get('nms_pre', -1), dtype=tf.int32)

        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                                 mlvl_bbox_preds,
                                                 mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score =tf.reshape(cls_score,(batch_size, -1,
                                                     self.cls_out_channels))
            if self.use_sigmoid_cls:
                scores = tf.math.sigmoid(cls_score)
            else:
                scores = tf.math.softmax(cls_score,axis=-1) 
            bbox_pred =tf.reshape(bbox_pred,(batch_size,-1,4)) 
            anchors =tf.broadcast_to(anchors,bbox_pred.shape)
            # Always keep topk op for dynamic input in onnx
        #     from mmdet.core.export import get_k_for_topk
        #     nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
        #     if nms_pre > 0:
        #         # Get maximum scores for foreground classes.
        #         if self.use_sigmoid_cls:
        #             max_scores, _ = scores.max(-1)
        #         else:
        #             # remind that we set FG labels to [0, num_class-1]
        #             # since mmdet v2.0
        #             # BG cat_id: num_class
        #             max_scores, _ = scores[..., :-1].max(-1)

        #         _, topk_inds = max_scores.topk(nms_pre)
        #         batch_inds = torch.arange(batch_size).view(
        #             -1, 1).expand_as(topk_inds)
        #         anchors = anchors[batch_inds, topk_inds, :]
        #         bbox_pred = bbox_pred[batch_inds, topk_inds, :]
        #         scores = scores[batch_inds, topk_inds, :]

        #     bboxes = self.bbox_coder.decode(
        #         anchors, bbox_pred, max_shape=img_shapes)
        #     mlvl_bboxes.append(bboxes)
        #     mlvl_scores.append(scores)

        # batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        # if rescale:
        #     batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
        #         scale_factors).unsqueeze(1)
        # batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)

        # # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        # if torch.onnx.is_in_onnx_export() and with_nms:
        #     from mmdet.core.export import add_dummy_nms_for_onnx
        #     # ignore background class
        #     if not self.use_sigmoid_cls:
        #         num_classes = batch_mlvl_scores.shape[2] - 1
        #         batch_mlvl_scores = batch_mlvl_scores[..., :num_classes]
        #     max_output_boxes_per_class = cfg.nms.get(
        #         'max_output_boxes_per_class', 200)
        #     iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        #     score_threshold = cfg.score_thr
        #     nms_pre = cfg.get('deploy_nms_pre', -1)
        #     return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
        #                                   max_output_boxes_per_class,
        #                                   iou_threshold, score_threshold,
        #                                   nms_pre, cfg.max_per_img)
        # if self.use_sigmoid_cls:
        #     # Add a dummy background class to the backend when using sigmoid
        #     # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        #     # BG cat_id: num_class
        #     padding = batch_mlvl_scores.new_zeros(batch_size,
        #                                           batch_mlvl_scores.shape[1],
        #                                           1)
        #     batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        # if with_nms:
        #     det_results = []
        #     for (mlvl_bboxes, mlvl_scores) in zip(batch_mlvl_bboxes,
        #                                           batch_mlvl_scores):
        #         det_bbox, det_label = multiclass_nms(mlvl_bboxes, mlvl_scores,
        #                                              cfg.score_thr, cfg.nms,
        #                                              cfg.max_per_img)
        #         det_results.append(tuple([det_bbox, det_label]))
        # else:
        #     det_results = [
        #         tuple(mlvl_bs)
        #         for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores)
        #     ]
        # return det_results
        tf.print("implement inf")