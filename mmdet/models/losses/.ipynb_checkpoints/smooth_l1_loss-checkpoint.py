
import tensorflow as tf

from ..builder import LOSSES
from .utils import weight_reduce_loss, weighted_loss

@weighted_loss
@tf.function(experimental_relax_shapes=True)
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
    Returns:
        torch.Tensor: Calculated loss
    """
    # assert beta > 0
    # assert pred.size() == target.size() and target.numel() > 0
    diff = tf.math.abs(pred - target)
    loss = tf.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return tf.math.reduce_sum(loss,axis=-1)
@weighted_loss
@tf.function(experimental_relax_shapes=True)
def l1_loss(pred, target):
    """L1 loss.
    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
    Returns:
        torch.Tensor: Calculated loss
    """
    
    loss = tf.math.abs(pred - target)
    return tf.math.reduce_sum(loss,axis=-1) 
@LOSSES.register_module()
class SmoothL1Loss(tf.keras.layers.Layer):
    """Smooth L1 loss.
    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
    @tf.function(experimental_relax_shapes=True)
    def call(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            )
        return loss_bbox


@LOSSES.register_module()
class L1Loss(tf.keras.layers.Layer):
    """L1 loss.
    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def call(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
#         print(pred.shape,target.shape,"focal")
        reduction = (
            reduction_override if reduction_override else self.reduction)
       
        # print(weight.shape)
        # a = tf.where(tf.reshape(weight,[-1,]) > 0)
        # a = tf.reshape(a,[-1,])
        # predx = tf.gather(pred,a)
        # targety =tf.gather(target,a)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
        # print(loss_bbox)
        # loss_bbox  = tf.math.abs(pred-target)
        
        # weight = tf.cast(weight,loss_bbox.dtype)
        # loss_bbox = tf.math.reduce_sum(loss_bbox,axis=-1) * tf.reshape(weight,(-1,))
        # print(loss_bbox)
        # return tf.math.reduce_sum(loss_bbox) / tf.cast(avg_factor,loss_bbox.dtype)
@LOSSES.register_module()
class BoxLoss():
  """L2 box regression loss."""

  def __init__(self, delta=0.1,is_exp=False, **kwargs):
    """Initialize box loss.
    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.is_exp=is_exp
    self.huber = tf.keras.losses.MeanAbsoluteError(
         reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def __call__(self, y_pred,
                y_true,
                weight=None,num_positives=None):
    normalizer = tf.cast(num_positives,y_pred.dtype)
    if self.is_exp:
        mask =tf.cast(tf.math.logical_not(tf.math.reduce_all(y_true == 0.0,axis=-1)), y_pred.dtype)
    else:
        mask = tf.cast(weight, y_pred.dtype)
    # TODO(fsx950223): remove cast when huber loss dtype is fixed.
    box_loss = tf.cast(self.huber(y_true, y_pred),
                       y_pred.dtype) * mask
    box_loss = tf.reduce_sum(box_loss) / normalizer
    return box_loss
