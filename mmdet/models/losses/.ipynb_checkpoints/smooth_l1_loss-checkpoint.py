
import tensorflow as tf

from ..builder import LOSSES
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
        print(pred.shape, target.shape, weight.shape,avg_factor,reduction_override)
        # assert reduction_override in (None, 'none', 'mean', 'sum')
        # reduction = (
        #     reduction_override if reduction_override else self.reduction)
        # loss_bbox = self.loss_weight * l1_loss(
        #     pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # return loss_bbox
        tf.print("debug l1 loss")
        return tf.constant([0.])