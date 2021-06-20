
from mmdet.models.dir_will_be_delete.mix_layers import SequentialLayer
from tensorflow import keras
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import pad
from ..builder import HEADS
from .anchor_head import AnchorHead
from ..dir_will_be_delete.norm import build_norm_layer
from ..dir_will_be_delete.conv import build_conv_layer
from ..dir_will_be_delete.conv_att_bn import ConvModule
@HEADS.register_module()
class RetinaHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.
    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.
    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)

    def m_init_layers(self):
        """Initialize layers of the head."""
        self.relu = tf.keras.layers.ReLU()# nn.ReLU(inplace=True)
        cls_convs = []
        reg_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(

                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.cls_convs = SequentialLayer(cls_convs)
        self.reg_convs = SequentialLayer(reg_convs)
        self.retina_cls =tf.keras.layers.Conv2D(self.num_anchors * self.cls_out_channels,3,padding='SAME')
        self.retina_reg = tf.keras.layers.Conv2D(self.num_anchors *4, 3 , padding='SAME')
    @tf.function(experimental_relax_shapes=True)
    def forward_single(self, x,training=False):
        """Forward feature of a single scale level.
        Args:
            x (Tensor): Features of a single scale level.
        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        print("call retinahead")
        cls_feat = x
        reg_feat = x
        cls_feat = self.cls_convs(x,training=training)
        reg_feat = self.reg_convs(x,training=training)
        cls_score = self.retina_cls(cls_feat,training=training)
        bbox_pred = self.retina_reg(reg_feat,training=training)
        return cls_score, bbox_pred