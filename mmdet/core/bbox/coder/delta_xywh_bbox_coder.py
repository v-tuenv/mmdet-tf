import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import mean

from ..builder import BBOX_CODERS
from .base_bbox_coder import BaseBBoxCoder
def fp16_clamp(x, min=None, max=None):
    
    if min is None and max is None:
        return x
    if min is not None and max is not None:
        return tf.clip_by_value(x, clip_value_min = min, clip_value_max = max)
    if min is not None:
        return tf.where(x < min, min, x)
    if max is not None:
        return tf.where(x > max, max , x)
    
    raise ValueError(min,max)

@BBOX_CODERS.register_module()
class DeltaXYWHBBoxCoder(BaseBBoxCoder):
    """Delta XYWH BBox coder.
    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        super(BaseBBoxCoder, self).__init__()
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            torch.Tensor: Box transformation deltas
        """
        encoded_bboxes = bbox2delta(bboxes, gt_bboxes, self.means, self.stds)
        return encoded_bboxes
    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.shape[0] == bboxes.shape[0]
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.shape[1] == bboxes.shape[1]
        decoded_bboxes = delta2bbox(bboxes, pred_bboxes, self.means, self.stds,
                                    max_shape, wh_ratio_clip, self.clip_border,
                                    self.add_ctr_clamp, self.ctr_clamp)

        return decoded_bboxes


def bbox2delta(proposals, gt, means=(0., 0., 0., 0.), stds=(1., 1., 1., 1.)):
    """Compute deltas of proposals w.r.t. gt.
    We usually compute the deltas of x, y, w, h of proposals w.r.t ground
    truth bboxes to get regression target.
    This is the inverse function of :func:`delta2bbox`.
    Args:
        proposals (Tensor): Boxes to be transformed, shape (N, ..., 4)
        gt (Tensor): Gt bboxes to be used as base, shape (N, ..., 4)
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
    Returns:
        Tensor: deltas with shape (N, 4), where columns represent dx, dy,
            dw, dh.
    """
    # assert proposals.shape == gt.shape

    proposals =tf.cast(proposals, tf.float32)# proposals.float()
    gt = tf.cast(gt, tf.float32) # gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0]
    ph = proposals[..., 3] - proposals[..., 1]

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0]
    gh = gt[..., 3] - gt[..., 1]

    dx = (gx - px) / pw 
    dy = (gy - py) / ph
    dw = tf.math.log(gw / pw)

    dh = tf.math.log(gh / ph)
    
    deltas = tf.stack([dx, dy, dw, dh], axis=-1)
    means = tf.convert_to_tensor(means, tf.float32)
    means = tf.expand_dims(means, 0)
    stds = tf.convert_to_tensor(stds)
    stds=tf.expand_dims(stds, 0)
   
    deltas =tf.math.subtract(deltas, means) 
    deltas = tf.math.divide(deltas, stds)
    return deltas

def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.
    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.
    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4), where 4 represent
           tl_x, tl_y, br_x, br_y.
    References:
        .. [1] https://arxiv.org/abs/1311.2524
    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    means = tf.convert_to_tensor(means, tf.float32)
    means  = tf.reshape(means, (1,-1))
    means = tf.tile(means, [1,deltas.shape[-1] // 4])
    stds = tf.convert_to_tensor(stds, tf.float32)
    stds  = tf.reshape(stds, (1,-1))
    stds = tf.tile(stds, [1,deltas.shape[-1] // 4])
    denorm_deltas = deltas * stds + means
    
    dx = denorm_deltas[..., 0]
    dy = denorm_deltas[..., 1]
    dw = denorm_deltas[..., 2]
    dh = denorm_deltas[..., 3]
    x1, y1 = rois[..., 0], rois[..., 1]
    x2, y2 = rois[..., 2], rois[..., 3]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5)
    py = ((y1 + y2) * 0.5)
    # px = tf.expand_dims(px, axis=-1)
    # py = tf.expand_dims(py, axis=-1)
    # px = tf.broadcast_to(px, dx.get_shape())
    # py = tf.broadcast_to(py, dy.get_shape())

    # Compute width/height of each roi
    pw = (x2 - x1)
    ph = (y2 - y1)
    # pw = tf.expand_dims(pw, axis=-1)
    # pw = tf.broadcast_to(pw, dw.get_shape())
    # ph = tf.expand_dims(ph, axis=-1)
    # ph = tf.broadcast_to(ph, dw.get_shape())
    dx_width = pw * dx
    dy_height = ph * dy

    max_ratio = np.abs(np.log(wh_ratio_clip))
    if add_ctr_clamp:
        dx_width = fp16_clamp(dx_width, max=ctr_clamp, min=-ctr_clamp)
        dy_height = fp16_clamp(dy_height, max=ctr_clamp, min=-ctr_clamp)
        dw = fp16_clamp(dw, max=max_ratio)
        dh = fp16_clamp(dh, max=max_ratio)
    else:
        dw = fp16_clamp(dw,min=-max_ratio, max=max_ratio)
        dh = fp16_clamp(dh,min=-max_ratio, max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * tf.math.exp(dw)
    gh = ph * tf.math.exp(dh)
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    bboxes = tf.stack([x1, y1, x2, y2], axis=-1)
    # bboxes = tf.reshape(bboxes, deltas.get_shape())
    

    if clip_border and max_shape is not None:
        
        # # clip bboxes with dynamic `min` and `max` for onnx
        
        if not isinstance(max_shape, tf.Tensor):
            max_shape = tf.convert_to_tensor(max_shape, dtype=x1.dtype)# x1.new_tensor(max_shape) 
        
        # if tf.rank(max_shape) == 2:
        #     assert bboxes.ndim == 3
        #     assert max_shape.size(0) == bboxes.size(0)

        min_xy =0.
        max_xy = tf.concat(
            [max_shape] * (deltas.shape[-1] // 2),
            axis=-1)
        max_xy = tf.reverse(max_xy, [-1])
        max_xy =tf.expand_dims(max_xy, axis=-2)
        bboxes = tf.where(bboxes < min_xy, min_xy, bboxes)
        bboxes = tf.where(bboxes > max_xy, max_xy, bboxes)

    return bboxes